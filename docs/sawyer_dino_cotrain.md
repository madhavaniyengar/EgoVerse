# Co-training with Sawyer Data via DINOv2 Visual Features

## Motivation

Paired robot transfer datasets (e.g. MimicGen) contain demonstrations for
multiple robot embodiments solving the same task. The robot arm data (Franka)
can be used directly for policy training. The non-target arm data (Sawyer) is
harder to use — its action space differs, and we have no calibrated end-effector
action labels that transfer to the target robot.

The core idea: instead of discarding the Sawyer data or trying to re-label its
actions in robot space, treat the Sawyer arm as a **surrogate human** and
supervise it with a representation that is embodiment-agnostic — the change in
visual scene content, captured by DINOv2 features.

## Method

### Action representation for the Sawyer arm

For each Sawyer episode, rather than using the raw 7D delta joint/EEF actions,
we extract the DINOv2-B CLS token from the agentview camera at every frame and
define the action at timestep `t` as:

```
action_dino[t] = CLS[t+1] - CLS[t]
```

This 768-dimensional vector encodes **what changes visually** when action `t` is
executed — a representation that is independent of the robot's kinematics or
morphology. The last frame uses a zero vector.

### Model architecture

A single shared transformer trunk processes observations from both embodiments.
Two domain-specific heads branch off the trunk:

| Head | Domain | Action space |
|---|---|---|
| Franka head | `franka_right_arm` | 7D cartesian EEF + gripper (xyz+ypr+gripper) × 100 steps |
| Sawyer head | `sawyer_as_human` | 768D DINOv2-B CLS delta × 100 steps |

Both heads use flow matching (Beta-time distribution, 50 inference steps) with a
`CrossTransformer` denoising network. The shared image encoder (ResNet → 256D)
and trunk (16-block, 256D, 8 heads) are trained jointly across both domains.

**Inputs per domain:**

| | Image | State |
|---|---|---|
| Franka | agentview RGB 128×128 → ResNet → 256D | xyz+ypr+gripper (7D) |
| Sawyer | agentview RGB 128×128 → shared ResNet → 256D | xyz+ypr (6D, no gripper) |

### Hypothesis

The Sawyer demonstrations solve the same manipulation task as the Franka
demonstrations and share the same camera viewpoint and object layout. By
training the shared trunk to predict coherent visual transitions (Sawyer head)
alongside robot actions (Franka head), the trunk learns a richer task-relevant
visual representation — one grounded in *what needs to happen* in the scene
rather than just robot kinematics. This should improve the Franka head's
sample efficiency and generalization.

## Data

Source: MimicGen paired pick-place dataset (`PickPlace_D0`), 32 workers.

| Split | Episodes | Frames (approx) |
|---|---|---|
| Franka (`franka_right_arm`) | 286 | ~190k |
| Sawyer (`sawyer_as_human`) | 286 | ~190k |

Zarr data stored at:
- `/project_data/held/madhavai/egoverse/mimicgen_pickplace_franka_zarr/`
- `/project_data/held/madhavai/egoverse/mimicgen_pickplace_sawyer_zarr/`

## Comparison

Two runs are trained to isolate the effect of the Sawyer co-training signal:

| Run | Config | Data |
|---|---|---|
| **Co-train** | `train_zarr_mimicgen_cotrain` | Franka + Sawyer (DINOv2 delta) |
| **Baseline** | `train_zarr_mimicgen_franka_only` | Franka only |

Both runs use an identical Franka head and trunk architecture, differing only in
whether the Sawyer head and data are included during training.

## Key files

| File | Purpose |
|---|---|
| `egomimic/scripts/custom_data/mimicgen_to_egoverse_zarr.py` | HDF5 → Zarr conversion; runs DINOv2-B inference offline |
| `egomimic/rldb/embodiment/custom.py` — `MimicgenSawyerHuman` | Keymap + transform list for the Sawyer domain |
| `egomimic/hydra_configs/model/hpt_cotrain_mimicgen.yaml` | Shared trunk + two-head model config |
| `egomimic/hydra_configs/train_zarr_mimicgen_cotrain.yaml` | Top-level co-train config |
| `egomimic/hydra_configs/train_zarr_mimicgen_franka_only.yaml` | Baseline config |
