# 05. Results, How to Present Them, and What to Claim

## 1. What should be shown to a reviewer or in a presentation

At this stage of the project, the strongest things to show are:

1. the **method diagram / narrative**,
2. the **dataset-to-pseudolabel pipeline**,
3. the fact that both Stage 1 and Stage 2 can train and checkpoint,
4. the **Stage 2 total loss curve** from the 500-step proof-of-concept run,
5. the run/debugging chronology showing the system was made operational end-to-end.

## 2. What should *not* be over-claimed

The current evidence supports these statements:

- the Architecture 1 training pipeline was implemented;
- it works on small subsets;
- Stage 1 and Stage 2 both reached training/checkpoint milestones;
- Stage 2 reached a successful 500-step proof-of-concept run;
- the training system logs meaningful losses and saves checkpoints.

The current evidence does **not** automatically support stronger claims such as:

- final high-quality edited 3D generation at production scale,
- strong quantitative evaluation across a large benchmark,
- robust generalisation across all edit types.

## 3. What the loss values mean here

The tail of the proof-of-concept Stage 2 run showed losses roughly in the ~1.2 to ~1.6 range for the total loss in late training steps.

That should be explained carefully:

- because the run is short and small-scale, the value is more a sign of **stable optimisation** than a sign of convergence;
- because the proof-of-concept training was performed on a very small working subset, the curve is mainly useful as engineering evidence;
- because there are also per-bin losses, the correct presentation curve should be the **overall `loss/loss` curve**.

## 4. What graph to include in the presentation

The main graph to include is:

- **smoothed total Stage 2 training loss vs step** using the TensorBoard tag `loss/loss`

Why this graph and not another one?

- it summarises the only fully successful longer Stage 2 run,
- it comes from the actual event file rather than hand-entered numbers,
- it avoids confusion from per-bin loss tags.

## 5. What the audience should be told about the 10-example setup

The presentation should make it explicit that:

- the system was prepared on a 100-sample subset,
- but the training/debugging proof-of-concept shown here used only 10 examples first.

This is an honest and technically sensible story. It tells the audience that the work was focused on **making the pipeline real and stable** before scaling it.

## 6. How to explain the 3D assets for the 10 examples

The best wording is:

> For each of the 10 working-set edit triplets, the edited target image was passed through the frozen TRELLIS 1 image-to-3D pipeline to obtain a pseudo-3D asset. That pseudo-3D asset was then encoded into the Stage 1 and Stage 2 latent targets used during fine-tuning.

This makes it clear that the project used TRELLIS itself as the pseudo-label generator.

## 7. Suggested presentation slide sequence

### Slide 1 — Problem

“2D edit triplets are abundant, but TRELLIS training expects 3D latent supervision.”

### Slide 2 — Solution idea

“Use frozen TRELLIS 1 as a pseudo-3D label factory.”

### Slide 3 — Data pipeline

- filtered Pix2Pix triplets,
- one-example validation,
- 100-sample subset,
- 10-example working subset,
- pseudo-3D asset creation,
- latent extraction.

### Slide 4 — Stage 1 / Stage 2 explanation

Explain why occupancy comes first and richer structured latent prediction comes second.

### Slide 5 — Engineering work

Summarise the sparse tensor / backend / checkpoint issues that had to be fixed.

### Slide 6 — Evidence of success

- checkpoint saves,
- successful 500-step Stage 2 run,
- TensorBoard total loss graph.

### Slide 7 — Limitations and next steps

Explain that the method works as a small-scale proof-of-concept and needs larger-scale training and cleaner preview/evaluation outputs next.

## 8. What a reviewer should conclude

A fair reviewer conclusion should be:

> The project successfully operationalised the Architecture 1 idea inside TRELLIS 1, demonstrated the full pseudolabel-to-latent-to-training path on small controlled subsets, and established a solid basis for scaling or evaluation in future work.

