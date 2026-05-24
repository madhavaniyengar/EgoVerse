"""Fit MANO to Aria's 21-keypoint hand data and render a side-by-side viz mp4.

Pipeline:
1. Pull first aria episode via SQL.
2. Build MultiDataset with Aria.get_keymap("keypoints") + transform_list
   "keypoints_headframe_ypr" -> "actions_keypoints" (left wrist+kp, right wrist+kp).
3. For the first N frames, stack aria keypoints into a batched tensor and fit
   MANO_RIGHT / MANO_LEFT to them in parallel (batched Adam).
4. Permute MANO's otaheri joint order (wrist, index*3, middle*3, pinky*3,
   ring*3, thumb*3, then 5 tips) to the canonical MANO ordering used by
   Human.FINGER_EDGES (wrist, thumb*4, index*4, middle*4, ring*4, pinky*4).
5. Render side-by-side: left half = Aria viz with Aria.FINGER_EDGES; right
   half = fitted MANO viz with Human (=Mecka) FINGER_EDGES.
6. Write mp4 to scratch/aria_to_mano.mp4.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import imageio_ffmpeg
import mano
import mediapy as mpy
import numpy as np
import torch

from egomimic.rldb.embodiment.human import Aria, Mecka
from egomimic.rldb.filters import DatasetFilter
from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset, S3EpisodeResolver
from egomimic.utils.aws.aws_data_utils import load_env
from egomimic.utils.aws.aws_sql import create_default_engine, episode_table_to_df

mpy.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())

REPO_ROOT = Path(__file__).resolve().parents[2].parent
SCRATCH_DIR = REPO_ROOT / "scratch"
CACHE_DIR = SCRATCH_DIR / "aria_to_mano_cache"
MANO_MODEL_DIR = REPO_ROOT / "external_ckpts" / "mano"
OUT_MP4 = SCRATCH_DIR / "aria_to_mano.mp4"
SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

NUM_FRAMES = 100
NUM_FIT_ITERS = 400
LR = 0.02
BETA_REG = 0.01

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# otaheri MANO joint order with return_tips=True (21 joints):
#   0 wrist, 1-3 index, 4-6 middle, 7-9 pinky, 10-12 ring, 13-15 thumb,
#   16 thumb_tip, 17 index_tip, 18 middle_tip, 19 ring_tip, 20 pinky_tip
# Canonical MANO order used by Human.FINGER_EDGES:
#   0 wrist, 1-4 thumb (CMC, MCP, IP, tip), 5-8 index, 9-12 middle,
#   13-16 ring, 17-20 pinky
OTAHERI_TO_CANONICAL = [
    0,              # wrist
    13, 14, 15, 16, # thumb (3 joints + tip)
    1, 2, 3, 17,    # index
    4, 5, 6, 18,    # middle
    10, 11, 12, 19, # ring
    7, 8, 9, 20,    # pinky
]

# Aria's 21-keypoint layout (read from Aria.FINGER_EDGES + user's note about idx 20):
#   0-4 = fingertips (thumb, index, middle, ring, pinky)
#   5 = palm root / wrist
#   6,7 = thumb intermediates  (Aria has 2; MANO has 3 incl. CMC)
#   8,9,10 = index intermediates (3, like MANO)
#   11,12,13 = middle intermediates
#   14,15,16 = ring intermediates
#   17,18,19 = pinky intermediates
#   20 = palm center (dropped from loss)
#
# Aria idx -> otaheri MANO target idx (per-point correspondence for the fit).
# We skip Aria's palm-center (20) and let MANO's thumb1/CMC float.
ARIA_TO_OTAHERI = {
    5: 0,    # wrist
    6: 14,   # thumb MCP    (Aria's 2 thumb mid joints -> MANO MCP+IP, skip CMC)
    7: 15,   # thumb IP
    0: 16,   # thumb tip
    8: 1,    # index1
    9: 2,    # index2
    10: 3,   # index3
    1: 17,   # index tip
    11: 4,   # middle1
    12: 5,   # middle2
    13: 6,   # middle3
    2: 18,   # middle tip
    14: 10,  # ring1
    15: 11,  # ring2
    16: 12,  # ring3
    3: 19,   # ring tip
    17: 7,   # pinky1
    18: 8,   # pinky2
    19: 9,   # pinky3
    4: 20,   # pinky tip
}
ARIA_IDX_USED = list(ARIA_TO_OTAHERI.keys())          # length 20
MANO_IDX_TARGET = [ARIA_TO_OTAHERI[a] for a in ARIA_IDX_USED]  # length 20


def fetch_episode_loader():
    load_env()
    engine = create_default_engine()
    df = episode_table_to_df(engine)
    aria_df = df[df["embodiment"].str.startswith("aria", na=False)]
    if len(aria_df) == 0:
        raise RuntimeError("No aria episodes found")
    episode_hash = aria_df.iloc[0]["episode_hash"]
    print(f"Aria episode: {episode_hash} (of {len(aria_df)})")

    key_map = Aria.get_keymap(keymap_mode="keypoints")
    transform_list = Aria.get_transform_list(mode="keypoints_headframe_ypr")
    resolver = S3EpisodeResolver(
        str(CACHE_DIR), key_map=key_map, transform_list=transform_list
    )
    filters = DatasetFilter(
        filter_lambdas=[f"lambda row: row['episode_hash'] == {episode_hash!r}"]
    )
    dataset = MultiDataset._from_resolver(
        resolver, filters=filters, sync_from_s3=True, mode="total"
    )
    return torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)


def gather_aria_keypoints(loader, n_frames):
    """Collect first n_frames worth of (image, left_kp, right_kp, batch) tuples.

    Aria's batched actions_keypoints layout (after keypoints_headframe_ypr):
      [left_wrist_xyz(3), left_wrist_ypr(3), left_kp(63), right_wrist_xyz(3),
       right_wrist_ypr(3), right_kp(63)] = 138 per timestep, shape (T_chunk, 138).
    We take chunk[0] (current timestep).
    """
    rows = []
    for i, batch in enumerate(loader):
        if i >= n_frames:
            break
        akp = batch["actions_keypoints"][0, 0]  # (138,)
        left_kp = akp[6:6 + 63].reshape(21, 3)
        right_kp = akp[75:75 + 63].reshape(21, 3)
        rows.append(
            {
                "batch": batch,
                "left_aria_kp": left_kp.float(),
                "right_aria_kp": right_kp.float(),
            }
        )
    print(f"Collected {len(rows)} frames")
    return rows


def fit_mano_to_aria_batched(aria_kp, is_rhand, n_iters, lr, beta_reg):
    """Batched MANO fit. aria_kp: (N, 21, 3). Returns canonical MANO kp (N, 21, 3)."""
    N = aria_kp.shape[0]
    aria_kp = aria_kp.to(DEVICE)

    # Per-frame validity (Aria stores invalid kps as NaN or implausible values).
    valid_mask = torch.isfinite(aria_kp).all(dim=-1)  # (N, 21)

    # Wrist init: aria[5] is wrist
    init_translation = torch.where(
        valid_mask[:, 5:6, None],
        aria_kp[:, 5:6, :],
        torch.zeros_like(aria_kp[:, 5:6, :]),
    ).squeeze(1)  # (N, 3)

    model = mano.load(
        model_path=str(MANO_MODEL_DIR),
        is_rhand=is_rhand,
        num_pca_comps=45,
        batch_size=N,
        flat_hand_mean=False,
    ).to(DEVICE)
    for p in model.parameters():
        p.requires_grad_(False)

    theta = torch.zeros(N, 45, device=DEVICE, requires_grad=True)
    beta = torch.zeros(N, 10, device=DEVICE, requires_grad=True)
    R_global = torch.zeros(N, 3, device=DEVICE, requires_grad=True)
    t_global = init_translation.detach().clone().requires_grad_(True)

    opt = torch.optim.Adam([theta, beta, R_global, t_global], lr=lr)

    # Targets: aria_kp[:, ARIA_IDX_USED, :] -> match against otaheri MANO[:, MANO_IDX_TARGET, :]
    aria_idx_t = torch.tensor(ARIA_IDX_USED, device=DEVICE, dtype=torch.long)
    mano_idx_t = torch.tensor(MANO_IDX_TARGET, device=DEVICE, dtype=torch.long)

    targets = aria_kp.index_select(1, aria_idx_t)  # (N, 20, 3)
    point_valid = torch.isfinite(targets).all(dim=-1)  # (N, 20)

    final_loss = torch.tensor(0.0)
    for step in range(n_iters):
        out = model(
            betas=beta, global_orient=R_global, hand_pose=theta,
            transl=t_global, return_verts=True, return_tips=True,
        )
        mano_kp = out.joints  # (N, 21, 3), otaheri ordering
        pred = mano_kp.index_select(1, mano_idx_t)  # (N, 20, 3)

        diff = (pred - targets) ** 2          # (N, 20, 3)
        per_point_err = diff.sum(dim=-1)      # (N, 20)
        masked = per_point_err * point_valid.float()
        denom = point_valid.sum().clamp(min=1).float()
        pos_loss = masked.sum() / denom
        reg_loss = beta_reg * (beta ** 2).sum() / N
        loss = pos_loss + reg_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 50 == 0 or step == n_iters - 1:
            print(f"    iter {step:4d}  pos={pos_loss.item():.5f}  reg={reg_loss.item():.5f}")
        final_loss = loss

    with torch.no_grad():
        out = model(
            betas=beta, global_orient=R_global, hand_pose=theta,
            transl=t_global, return_verts=True, return_tips=True,
        )
        mano_kp = out.joints.detach().cpu()  # (N, 21, 3) otaheri order

    perm = torch.tensor(OTAHERI_TO_CANONICAL, dtype=torch.long)
    canonical = mano_kp.index_select(1, perm)  # (N, 21, 3) canonical order

    # If a frame's wrist was invalid, blank that frame's keypoints with NaN.
    frame_valid = valid_mask[:, 5].cpu()  # wrist valid?
    canonical = torch.where(
        frame_valid[:, None, None], canonical, torch.full_like(canonical, float("nan"))
    )
    return canonical


def render_side_by_side(rows, mano_left_canonical, mano_right_canonical):
    """Render aria-viz | mano-viz side-by-side per frame."""
    frames = []
    for i, row in enumerate(rows):
        batch = row["batch"]
        aria_vis = Aria.viz_transformed_batch(
            batch, mode="keypoints", viz_batch_key="actions_keypoints"
        )

        # Build MANO viz tensor (126,) = left_21*3 + right_21*3, no wrist prefix.
        left_flat = mano_left_canonical[i].reshape(-1).numpy()
        right_flat = mano_right_canonical[i].reshape(-1).numpy()
        viz_data = np.concatenate([left_flat, right_flat])  # (126,)
        # Aria episodes use intrinsics_key "base"
        image_t = batch[Aria.VIZ_IMAGE_KEY][0]
        if image_t.ndim == 3 and image_t.shape[0] in (1, 3):
            image_t = image_t.permute(1, 2, 0)
        image = image_t.numpy()
        if image.dtype != np.uint8:
            image = (image * 255.0).clip(0, 255).astype(np.uint8)
        mano_vis = Mecka.viz(
            image=image, viz_data=viz_data, mode="keypoints", intrinsics_key="base"
        )

        h = max(aria_vis.shape[0], mano_vis.shape[0])
        def pad(im):
            if im.shape[0] != h:
                pad_amt = h - im.shape[0]
                im = cv2.copyMakeBorder(im, 0, pad_amt, 0, 0, cv2.BORDER_CONSTANT)
            return im
        a = pad(aria_vis); m = pad(mano_vis)
        # Labels
        cv2.putText(a, "Aria (original)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(m, "Aria->MANO (fit, canonical order)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        combined = np.concatenate([a, m], axis=1)
        frames.append(combined)
    return frames


def main() -> None:
    loader = fetch_episode_loader()
    rows = gather_aria_keypoints(loader, NUM_FRAMES)

    aria_left = torch.stack([r["left_aria_kp"] for r in rows])    # (N, 21, 3)
    aria_right = torch.stack([r["right_aria_kp"] for r in rows])  # (N, 21, 3)
    print(f"Left aria valid wrist count: {torch.isfinite(aria_left[:,5]).all(-1).sum().item()}/{len(rows)}")
    print(f"Right aria valid wrist count: {torch.isfinite(aria_right[:,5]).all(-1).sum().item()}/{len(rows)}")

    print("Fitting LEFT hand...")
    mano_left = fit_mano_to_aria_batched(aria_left, is_rhand=False, n_iters=NUM_FIT_ITERS, lr=LR, beta_reg=BETA_REG)
    print("Fitting RIGHT hand...")
    mano_right = fit_mano_to_aria_batched(aria_right, is_rhand=True, n_iters=NUM_FIT_ITERS, lr=LR, beta_reg=BETA_REG)

    print("Rendering...")
    frames = render_side_by_side(rows, mano_left, mano_right)
    mpy.write_video(str(OUT_MP4), frames, fps=30)
    print(f"Wrote {OUT_MP4} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
