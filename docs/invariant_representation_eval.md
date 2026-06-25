# Evaluation Plan for Cycle-Aligned Panda/Sawyer HPT

## Summary

Evaluate the model at two levels:

- **Representation eval:** frozen-checkpoint analysis on held-out Panda Square and Sawyer batches, comparing the cycle-aligned model against the prior co-trained no-alignment checkpoint.
- **Policy eval:** first use existing offline action/video eval, then add a simulator rollout evaluator for success rate on MimicGen/robosuite-style Square.

Use these datasets:

- Panda: `/home/ubuntu/data/square_d0_panda_controller_zarr_200`
- Sawyer: `/home/ubuntu/data/mimicgen_pickplace_sawyer_zarr`

Use fresh norm stats for this pairing, not the old Franka/Sawyer stats.

## Representation Eval

Add a new evaluator/script, for example `egomimic/eval/eval_representation_alignment.py`, plus a Hydra config.

For each checkpoint, extract frozen trunk representations using the same HPT pathway as training:

- Build batches through the existing datamodule and `process_batch_for_training`.
- Convert each domain with `_robomimic_to_hpt_data`.
- Call `algo.nets["policy"].forward_features(domain, data)`.
- Use the final action-token/global feature as the primary representation.
- Optionally save per-layer/block features for diagnosing where alignment emerges.

Metrics to compute:

- **Cycle diagnostics:** cycle reconstruction MSE, mapped-nearest-neighbor MSE, variance/effective-rank, and collapse checks.
- **Proxy-conditioned retrieval:** use normalized shared EE pose proxy, dims `0:6`, to define comparable Panda/Sawyer samples. Report Recall@1/5/10, MRR, and mean proxy distance of retrieved neighbors.
- **Action-conditioned alignment:** only score retrieval among samples whose proxy distance is within a fixed radius or top-k proxy candidates, matching the training alignment condition.
- **Domain invariance:** train a frozen-feature linear domain classifier. Lower accuracy is better, but report it alongside task-retention metrics to avoid rewarding collapse.
- **Task retention:** from frozen features, linearly regress the shared EE proxy and report MSE/R2. This guards against aligned but useless features.
- **CKA / cosine alignment:** compute CKA and cosine similarity on proxy-matched Panda/Sawyer pairs.

Artifacts:

- `representation_metrics.json`
- `embeddings.npz` with checkpoint name, domain, episode/frame ids, proxy, and embedding
- UMAP/t-SNE plots colored by domain and proxy bins
- Retrieval examples: nearest Panda/Sawyer image pairs with proxy distance and representation distance

Compare:

- cycle-aligned checkpoint
- prior co-trained no-alignment checkpoint

## Offline Action Eval

Reuse the existing `HPTEvalVideo` path before simulator rollout.

Run each checkpoint in `mode=eval` with:

- `paths.mimicgen_franka_dataset_dir=/home/ubuntu/data/square_d0_panda_controller_zarr_200`
- same Sawyer path
- fresh/precomputed norm stats from the corrected Panda/Sawyer run
- `evaluator.write_videos=false` for metrics-only, then true for a small qualitative pass

Report existing metrics:

- `Valid/action_loss`
- Panda `actions_cartesian` paired/final MSE and Frechet metrics
- Sawyer `actions_dino` paired/final MSE and Frechet metrics
- validation videos for a small fixed subset

This is not rollout success, but it is the sanity check that the policy heads still predict plausible actions.

## Simulator Rollout Eval

Add a MimicGen/robosuite rollout evaluator because the repo does not currently contain a complete sim rollout runner for these checkpoints.

Implement a script/config that:

- Loads a checkpoint via `ModelWrapper.load_from_checkpoint`.
- Creates the corresponding MimicGen Square/PickPlace robosuite environment using the source task metadata when available.
- Runs deterministic seeds, fixed initial states, and fixed camera observations.
- Converts sim observations into the same batch schema used by `MimicgenFranka`.
- Calls `model.forward_eval` for the Panda/Franka head.
- Executes actions in receding horizon mode: query every `N` sim steps, execute the first chunk or a downsampled prefix.
- Records success from the environment's native success condition, plus video and trajectory logs.

Primary rollout metrics:

- success rate over seeds
- mean steps-to-success
- timeout/failure reason counts
- action smoothness and command magnitude
- per-seed video

Recommended protocol:

- 50 seeds for quick comparison
- 200 seeds for final numbers
- same seeds for cycle-aligned and no-alignment baseline
- evaluate Panda Square first, since the Panda dataset is Square and the policy head emits real robot/cartesian actions

## Test Plan

- Unit test representation extraction on one Panda batch and one Sawyer batch; assert finite embeddings and expected shapes.
- Unit test retrieval metrics on synthetic embeddings where correct matches are known.
- Smoke test representation eval with `limit_batches=2`.
- Smoke test offline `mode=eval` with `limit_val_batches=2`.
- Smoke test sim rollout with one seed and short horizon, checking action conversion and video writing.

Acceptance criteria:

- no stale Franka norm stats are used
- both checkpoints evaluate on identical batches/seeds
- representation metrics include collapse guards
- rollout evaluator produces `rollout_metrics.json` and videos

## Assumptions

- Rollout eval means simulator-first evaluation, not real-robot hardware yet.
- Main comparison is cycle-aligned checkpoint vs prior co-trained no-alignment checkpoint.
- Primary representation is the final shared trunk/action-token feature from `forward_features`.
- Shared action similarity is approximated with normalized EE pose proxy because Panda real actions and Sawyer DINO-delta actions are not directly comparable.
