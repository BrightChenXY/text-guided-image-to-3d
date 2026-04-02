# 04. Arch1 Runs and the Debugging Timeline

This file is intentionally practical. It explains not just the clean method, but the actual engineering path taken to make the method run.

## 1. Stage 1 smoke training reached a working checkpoint state

A successful Stage 1 smoke run produced a checkpoint directory with the expected files, including:

- `denoiser_step0000025.pt`
- `denoiser_step0000050.pt`
- EMA versions,
- `text_proj_step0000025.pt`
- `text_proj_step0000050.pt`
- `misc_step0000025.pt`
- `misc_step0000050.pt`

This was the first clear sign that the Architecture 1 Stage 1 training path was truly functioning.

## 2. One early Stage 1 bug: `sample_id` leaking into the model call

During Stage 1, an error occurred because the dataset/trainer path passed `sample_id` into the model forward call, but the model forward function did not accept it.

That was fixed by removing `sample_id` from the minibatch content passed into the model.

This seems small, but it was a classic symptom that the dataset schema and the model API had not yet been perfectly aligned.

## 3. Stage 2 introduced sampler-related dataset requirements

When Stage 2 was first attempted, the `BalancedResumableSampler` required the dataset to expose a `loads` attribute. The dataset did not initially provide it, causing an assertion failure.

That was fixed by patching the Stage 2 dataset loader to compute and expose `loads` based on the sparse latent target voxel count.

This was an important moment because it revealed that Stage 2 training had stricter dataset-side requirements than Stage 1.

## 4. Sparse tensor transfer and wrapper failures

A major portion of the debugging effort went into sparse tensor compatibility.

The failure sequence included errors such as:

- `NoneType` object is not callable,
- tensors split across CPU and CUDA devices,
- missing sparse tensor attributes such as `features`, `grid`, `is_quantized`, `spatial_shape`, `shadow_copy`, `find_indice_pair`, `_timer`, and `_indices`.

This tells the reviewer something very important:

> The architecture itself was reasonable, but the engineering path was blocked by incompatibilities between TRELLIS sparse wrappers and the assumptions made by the downstream sparse backend stack.

The debugging work therefore focused on making the project’s sparse tensor objects behave in a backend-compatible way during `.to(device)`, replacement, and spconv execution.

## 5. Backend and dependency issues

The session also surfaced environment-level problems, including:

- xformers / flash attention backend switching,
- missing `flash_attn`,
- malformed `try/except` edits during patching,
- missing `easydict` in one environment invocation,
- differences between running inside the intended conda environment and outside it.

These issues were part of the reason the work repeatedly returned to smoke-style validation rather than immediately jumping to a polished inference demo.

## 6. Checkpoint save/load issues

Another engineering theme was checkpoint robustness.

Observed issues included:

- save failures during checkpoint writing,
- empty checkpoint directories in earlier Stage 2 proof-of-concept attempts,
- corrupted `misc_step...pt` archives,
- one load path trying to call `convert_to_fp16()` on `TextImageConditionProjector`, which did not support it.

These are important details for a reviewer because they explain why not every run directory that exists should be treated as a valid training result.

## 7. Which Stage 2 run should be treated as the main proof-of-concept run

The clearest successful Stage 2 run in this session is:

- `/workspace/arch1_runs/stage2_poc_500_fresh`

That run:

- completed training to **500 steps**,
- saved checkpoints at **250** and **500**,
- produced a TensorBoard event file,
- and later supported loss-curve extraction.

By contrast, an earlier `stage2_poc_500` directory had an effectively empty `ckpts/` folder and therefore should **not** be treated as the main preserved run.

## 8. Stage 2 preview/snapshot attempts

After the 500-step run, there were attempts to create preview outputs from the checkpoint. Those attempts ran into two practical issues:

1. the Architecture 1 trainer mixin had originally overridden snapshot methods so that smoke training skipped snapshots;
2. re-enabling snapshot paths exposed assumptions inside snapshot utilities that did not cleanly handle sparse tensors as image-like tensors.

So the key output of the project at this stage is **training evidence and checkpoint evidence**, not yet a polished automatic preview pipeline.

## 9. Loss extraction and graph generation

The TensorBoard event file for the successful Stage 2 proof-of-concept run contained many scalar tags, including:

- `loss/mse`
- `loss/loss`
- several `loss/bin_X/mse` tags

An initial plotting attempt mistakenly picked a per-bin loss tag such as `loss/bin_8/mse`. The correct reviewer-facing curve should instead use the **total training loss tag**:

- `loss/loss`

A smoothed total loss curve was then generated and saved as:

- `loss_curve_total_smoothed.png`

This is the graph that should be shown in a presentation, not the per-bin curve.

## 10. What this debugging history proves

The debugging history proves that the project was not just a design exercise. It reached the level where:

- preprocessing logic existed,
- stage-aware loaders existed,
- checkpoints were produced,
- successful reloading was achieved for key runs,
- TensorBoard losses were logged,
- and a total-loss plot could be generated.

That is strong evidence of a working proof-of-concept engineering pipeline.

