"""Serve an EgoVerse HPT checkpoint to the PolaRiS evaluation client.

All image preprocessing lives here so evaluation exactly matches dataset creation:
INTER_AREA downsampling, centered fixed-camera crops, and an explicit wrist crop.
"""

from __future__ import annotations

import argparse
import pickle
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch
import zmq
from scipy.spatial.transform import Rotation

import egomimic.utils.hydra_resolvers  # noqa: F401
from egomimic.pl_utils.pl_model import ModelWrapper

DOMAIN = "franka_right_arm"
EMBODIMENT_ID = 16
ACTION_KEY = "actions_cartesian"
STATE_KEY = "observations.state.ee_pose"
CAMERA_KEYS = ("front_img_1", "wrist_img")


def _load_transform(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        transform = np.load(path)
    else:
        transform = np.loadtxt(path)
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 transform at {path}, got {transform.shape}")
    if not np.isfinite(transform).all():
        raise ValueError(f"Transform at {path} contains NaN/Inf")
    if not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-8):
        raise ValueError(f"Transform at {path} is not homogeneous")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4):
        raise ValueError(f"Transform at {path} has a non-orthonormal rotation")
    return transform


def _transform_xyzypr(poses: np.ndarray, target_from_source: np.ndarray) -> np.ndarray:
    """Transform xyz+ZYX-YPR+gripper poses between rigid coordinate frames."""
    poses = np.asarray(poses)
    if poses.shape[-1] != 7:
        raise ValueError(f"Expected (..., 7) xyz+ypr+gripper poses, got {poses.shape}")
    flat = poses.reshape(-1, 7).astype(np.float64)
    source_from_eef = Rotation.from_euler("ZYX", flat[:, 3:6]).as_matrix()
    target_from_eef = target_from_source[:3, :3][None] @ source_from_eef
    xyz = (
        target_from_source[:3, :3] @ flat[:, :3].T
    ).T + target_from_source[:3, 3]
    ypr = Rotation.from_matrix(target_from_eef).as_euler("ZYX")
    result = np.concatenate([xyz, ypr, flat[:, 6:7]], axis=-1)
    return result.reshape(poses.shape).astype(np.float32)


def _preprocess_image(
    image: np.ndarray,
    *,
    scale_factor: int,
    crop_shape: tuple[int, int],
    crop_left: int | None,
) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"Expected HWC RGB image, got {arr.shape}")
    if scale_factor < 1:
        raise ValueError(f"scale_factor must be >= 1, got {scale_factor}")
    if scale_factor > 1:
        h, w = arr.shape[:2]
        arr = cv2.resize(
            arr,
            (w // scale_factor, h // scale_factor),
            interpolation=cv2.INTER_AREA,
        )
    crop_h, crop_w = crop_shape
    h, w = arr.shape[:2]
    if crop_h > h or crop_w > w:
        raise ValueError(f"Crop {crop_shape} does not fit scaled image {h}x{w}")
    top = (h - crop_h) // 2
    left = (w - crop_w) // 2 if crop_left is None else int(crop_left)
    if not 0 <= left <= w - crop_w:
        raise ValueError(
            f"crop_left={left} outside [0, {w - crop_w}] for image {h}x{w} and crop {crop_shape}"
        )
    return np.ascontiguousarray(arr[top : top + crop_h, left : left + crop_w])


def _resample_chunk(chunk: np.ndarray, target_length: int) -> np.ndarray:
    if target_length <= 0 or len(chunk) == target_length:
        return chunk.astype(np.float32, copy=False)
    old_t = np.linspace(0.0, 1.0, len(chunk))
    new_t = np.linspace(0.0, 1.0, target_length)
    out = np.empty((target_length, 7), dtype=np.float64)
    for dim in (0, 1, 2, 6):
        out[:, dim] = np.interp(new_t, old_t, chunk[:, dim])
    angles = np.unwrap(chunk[:, 3:6], axis=0)
    for dim in range(3):
        out[:, dim + 3] = np.interp(new_t, old_t, angles[:, dim])
    out[:, 3:6] = (out[:, 3:6] + np.pi) % (2 * np.pi) - np.pi
    return out.astype(np.float32)


def _load_checkpoint(path: Path, device: torch.device) -> ModelWrapper:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    hparams = checkpoint["hyper_parameters"]
    wrapper = ModelWrapper(
        config_tree=hparams["config_tree"],
        norm_stats_state=hparams["norm_stats_state"],
        scheduler_interval=hparams.get("scheduler_interval", "step"),
        scheduler_frequency=hparams.get("scheduler_frequency", 1),
        enable_grad_norm=hparams.get("enable_grad_norm", False),
    )
    wrapper.load_state_dict(checkpoint["state_dict"], strict=True)
    wrapper.eval().to(device)
    wrapper.model.device = device
    wrapper.model.nets["policy"].device = device
    return wrapper


class EgoVerseInference:
    def __init__(
        self,
        checkpoint: Path,
        device: str,
        image_scale_factor: int,
        crop_shape: tuple[int, int],
        wrist_crop_left: int,
        camera_from_base: Path,
        resampled_action_len: int,
        num_inference_steps: int,
        crop_viz_output: Path | None,
    ) -> None:
        self.device = torch.device(device)
        self.wrapper = _load_checkpoint(checkpoint, self.device)
        self.algo = self.wrapper.model
        self.policy = self.algo.nets["policy"]
        self.image_scale_factor = image_scale_factor
        self.crop_shape = crop_shape
        self.wrist_crop_left = wrist_crop_left
        self.camera_from_base = _load_transform(camera_from_base)
        self.base_from_camera = np.linalg.inv(self.camera_from_base)
        self.resampled_action_len = resampled_action_len
        self.crop_viz_output = crop_viz_output
        self._wrote_preview = False

        for head in self.policy.heads.values():
            if hasattr(head, "num_inference_steps"):
                head.num_inference_steps = num_inference_steps

        configured = {key.rsplit(".", 1)[-1] for key in self.algo.camera_keys[EMBODIMENT_ID]}
        encoded = set(self.algo.encoders.keys())
        if not set(CAMERA_KEYS).issubset(configured) or not set(CAMERA_KEYS).issubset(encoded):
            raise ValueError(
                f"Checkpoint does not consume required cameras {CAMERA_KEYS}: "
                f"configured={configured}, encoders={encoded}"
            )
        print(
            f"[EgoVerse] checkpoint={checkpoint} cameras={CAMERA_KEYS} "
            f"scale={image_scale_factor} crop={crop_shape} wrist_left={wrist_crop_left}"
        )

    def _images(self, request: dict) -> dict[str, np.ndarray]:
        raw = request["images"]
        images = {
            "front_img_1": _preprocess_image(
                raw["front_img_1"], scale_factor=self.image_scale_factor,
                crop_shape=self.crop_shape, crop_left=None,
            ),
            "wrist_img": _preprocess_image(
                raw["wrist_img"], scale_factor=self.image_scale_factor,
                crop_shape=self.crop_shape, crop_left=self.wrist_crop_left,
            ),
        }
        if self.crop_viz_output is not None and not self._wrote_preview:
            preview = np.concatenate([images[key] for key in CAMERA_KEYS], axis=1)
            self.crop_viz_output.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(self.crop_viz_output), cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))
            self._wrote_preview = True
            print(f"[EgoVerse] wrote crop preview: {self.crop_viz_output}")
        return images

    def _image_tensor(self, image: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(image).permute(2, 0, 1).float().div_(255.0)
        return tensor.unsqueeze(0).to(self.device)

    @torch.inference_mode()
    def infer(self, request: dict) -> dict:
        images = self._images(request)
        state = np.asarray(request["state_ee_pose"], dtype=np.float32)
        if state.shape != (7,):
            raise ValueError(f"state_ee_pose must be xyz+ypr+gripper (7,), got {state.shape}")
        state = _transform_xyzypr(state, self.camera_from_base)

        raw_batch = {
            f"observations.images.{key}": self._image_tensor(value)
            for key, value in images.items()
        }
        raw_batch[STATE_KEY] = torch.from_numpy(state).view(1, 7).to(self.device)
        norm_batch = self.algo.norm_stats.normalize(raw_batch, EMBODIMENT_ID)

        data = {"state_ee_pose": norm_batch[STATE_KEY].unsqueeze(1)}
        for key in CAMERA_KEYS:
            full_key = f"observations.images.{key}"
            image = norm_batch[full_key]
            if self.algo.eval_image_augs:
                image = self.algo.eval_image_augs(image)
            data[key] = image.unsqueeze(1).unsqueeze(1)
        data.update(
            is_6dof=self.algo.is_6dof,
            pad_mask=torch.ones(1, 100, 1, device=self.device),
            embodiment=torch.tensor([EMBODIMENT_ID], dtype=torch.int64, device=self.device),
            action=torch.zeros(1, 100, 7, device=self.device),
        )
        prediction = self.policy.forward(DOMAIN, data)[DOMAIN]
        prediction = self.algo.norm_stats.unnormalize(
            {ACTION_KEY: prediction}, EMBODIMENT_ID
        )[ACTION_KEY]
        chunk = prediction.squeeze(0).float().cpu().numpy()
        chunk = _resample_chunk(chunk, self.resampled_action_len)
        chunk = _transform_xyzypr(chunk, self.base_from_camera)
        if not np.isfinite(chunk).all():
            raise ValueError("Model returned non-finite actions")

        horizon = max(1, min(int(request.get("open_loop_horizon", 1)), len(chunk)))
        viz = np.concatenate([images[key] for key in CAMERA_KEYS], axis=1)
        # Keep NumPy objects out of the pickle wire format. The server and
        # PolaRiS client can use different NumPy major versions (NumPy 2 uses
        # the private ``numpy._core`` module, which NumPy 1 cannot unpickle).
        viz = np.ascontiguousarray(viz)
        return {
            "action_chunk": chunk[:horizon].tolist(),
            "viz_bytes": viz.tobytes(),
            "viz_shape": viz.shape,
            "viz_dtype": viz.dtype.str,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5557)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--image-scale-factor", type=int, default=2)
    parser.add_argument("--crop-shape", type=int, nargs=2, default=(360, 480), metavar=("H", "W"))
    parser.add_argument("--wrist-crop-left", type=int, default=160)
    parser.add_argument(
        "--camera-from-base",
        type=Path,
        required=True,
        help="4x4 .npy/txt transform used during training: T_camera_base",
    )
    parser.add_argument("--resampled-action-len", type=int, default=45)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--crop-viz-output", type=Path, default=Path("/tmp/egoverse_crop_preview.jpg"))
    args = parser.parse_args()

    inference = EgoVerseInference(
        checkpoint=args.checkpoint,
        device=args.device,
        image_scale_factor=args.image_scale_factor,
        crop_shape=tuple(args.crop_shape),
        wrist_crop_left=args.wrist_crop_left,
        camera_from_base=args.camera_from_base,
        resampled_action_len=args.resampled_action_len,
        num_inference_steps=args.num_inference_steps,
        crop_viz_output=args.crop_viz_output,
    )
    context = zmq.Context.instance()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://{args.host}:{args.port}")
    print(f"EgoVerse server listening on tcp://{args.host}:{args.port}", flush=True)
    while True:
        try:
            request = pickle.loads(socket.recv())
            if request.get("reset"):
                socket.send(pickle.dumps({"status": "ok"}))
            else:
                socket.send(pickle.dumps(inference.infer(request)))
        except Exception as exc:
            traceback.print_exc()
            socket.send(pickle.dumps({"error": f"{type(exc).__name__}: {exc}"}))


if __name__ == "__main__":
    main()
