# 03. Why There Are Two Stages, and How Training Was Done

## 1. Why TRELLIS 1 training is split into Stage 1 and Stage 2

TRELLIS 1 does not directly jump from conditioning to the final textured 3D result in one step.

Instead, it has a two-stage latent prediction structure:

- **Stage 1** predicts sparse 3D structure / occupancy,
- **Stage 2** predicts the structured latent that carries richer geometry and appearance information.

The architecture report explains this split directly:

- Stage 1 predicts the sparse structure target (`z_ss_target`),
- Stage 2 predicts the structured latent target (`z_slat_target`) and uses the Stage 1 occupancy information as part of its input path.

## 2. Why this split matters conceptually

This two-stage split matters because it decomposes the problem:

### Stage 1 answers:

> Where does the object exist in 3D space?

### Stage 2 answers:

> Given that sparse structure, what are the richer shape and appearance latents?

That is why the project could not realistically skip Stage 1 and train only Stage 2 from scratch for a faithful TRELLIS-style fine-tune.

## 3. What changed relative to original TRELLIS 1

The architecture report emphasises that the **training objective did not fundamentally change**. The important changes were:

1. the conditioning is now **image + text** rather than only image;
2. the DiTs are **initialised from released TRELLIS checkpoints** instead of being trained from scratch;
3. the targets are **pseudo-label latents** derived from edited target images.

That means this project kept the TRELLIS training logic intact, while changing the source of supervision and the conditioning signal.

## 4. Conditioning path used in this project

The architecture design note states that the conditioning fusion is:

- frozen DINOv2 image features from the source image,
- frozen CLIP text features from the edit prompt,
- a trainable linear projection from 768 → 1024 for the text tokens,
- token concatenation to form `e_joint`.

So the project was not trying to invent a completely new attention stack. It used TRELLIS’s existing conditioning mechanism and made the conditioning sequence richer.

## 5. What is frozen and what is trainable

Per the architecture note, the intended trainable/frozen split is:

### Frozen

- SS-VAE encoder/decoder,
- SLAT-VAE encoder/decoders,
- DINOv2 image encoder,
- CLIP text encoder.

### Trainable

- the text projection layer,
- Stage 1 DiT,
- Stage 2 DiT.

That is exactly the kind of fine-tuning setup that makes sense for this proof-of-concept: preserve TRELLIS’s latent space and decoder ecosystem, but adapt the latent predictors to the new edit-conditioned task.

## 6. How Stage 1 was trained in this session

Stage 1 corresponds to the sparse structure flow model.

In the runs shown during this project session:

- the Stage 1 dataset loader was built around the Architecture 1 conditioned dataset;
- the conditioning path included both source-image and text embeddings;
- early runs were performed as **smoke tests** to prove that training, loss computation, logging, and checkpoint saving worked;
- a successful Stage 1 smoke run saved checkpoints at step **25** and **50**;
- a larger Stage 1 run was also brought to the point where a **500-step checkpoint** could be loaded successfully later.

This matters because Stage 1 was not merely conceptual. It reached a state where checkpoints existed and could be reloaded, which is a concrete sign that the training path was functioning.

## 7. How Stage 2 was trained in this session

Stage 2 corresponds to the structured latent flow model.

Stage 2 was significantly harder to stabilise because it involved:

- sparse tensor movement to GPU,
- elastic memory control,
- balanced resumable sampling,
- spconv/xformers/attention backend compatibility,
- checkpoint loading and saving,
- and many wrapper assumptions around sparse tensor objects.

Nevertheless, after multiple rounds of debugging and patching, Stage 2 reached a successful proof-of-concept state:

- a smoke-train path was made to run end-to-end,
- and later the full proof-of-concept directory `stage2_poc_500_fresh` reached **500 steps** successfully,
- saving checkpoints at **250** and **500**.

## 8. Why the smoke runs were necessary

The smoke runs were not only about “small training.” They were used to verify a chain of engineering assumptions:

- the dataset loader returns the right keys and shapes;
- the model forward signature matches the dataset outputs;
- the sparse tensor objects can survive device transfer;
- the samplers can batch the examples;
- the training loop can save checkpoints;
- the checkpoint loader can resume later.

In a project like this, that sequence is a real part of the research contribution, because without it the architecture note remains only a design document.

## 9. Why Stage 1 and Stage 2 had to be explained to reviewers separately

A reviewer needs to understand that the project had two different notions of “working”:

- **Stage 1 working** means the sparse structure predictor can train and save checkpoints.
- **Stage 2 working** means the richer structured latent predictor can train on sparse inputs without backend and sparse-wrapper failures.

Stage 2 is therefore the more difficult engineering milestone.

## 10. Reviewer summary

The reviewer-level takeaway is:

> The project did not simply say “we fine-tuned TRELLIS.” It implemented the TRELLIS 1 two-stage fine-tuning logic in the Architecture 1 setting, first bringing Stage 1 to a stable checkpointable state, then fighting through sparse/backend/debugging issues until Stage 2 could run a successful 500-step proof-of-concept training run.

