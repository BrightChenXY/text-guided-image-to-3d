# 07. Limitations and Next Steps

## 1. Main limitations of the current project state

### Small-scale training only

The proof-of-concept runs were intentionally small. That was the correct engineering decision, but it means the current evidence is mainly about **pipeline validity**, not final model quality.

### Pseudo-label quality is bounded by frozen TRELLIS 1

If frozen TRELLIS reconstruction of the edited target image is poor, the training target will also be poor.

### Preview/inference demonstration is less mature than training demonstration

The project reached strong evidence on training/checkpointing, but automatic snapshot/preview export remained less polished because of sparse-tensor assumptions inside snapshot utilities.

### Sparse backend complexity was a major bottleneck

A large fraction of the work went into compatibility rather than pure model design.

## 2. What the most logical next steps are

### Scale from 10 to larger subsets safely

The immediate next step is not “change the whole method.” It is to keep the method fixed and scale the subset size carefully beyond the tiny proof-of-concept setting.

### Produce a clean end-to-end inference demo from saved checkpoints

That would give the project a qualitative result page with before/after and 3D outputs.

### Improve artifact preservation

If not already done, preserve:

- selected checkpoints,
- the total-loss figure,
- filtered subset manifests,
- and a clean run table.

### Add qualitative sample sheets

For a reviewer, side-by-side sheets showing:

- source image,
- edit prompt,
- target edited image,
- pseudo-3D asset,
- and final inferred result,

would make the project much easier to evaluate.

## 3. Reviewer-level final conclusion

This work should be presented as a **successful proof-of-concept implementation of Architecture 1 in TRELLIS 1**, with major practical debugging completed and a clear path to larger-scale training and cleaner qualitative demonstration.

