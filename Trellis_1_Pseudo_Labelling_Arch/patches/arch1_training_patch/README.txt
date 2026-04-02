Architecture-1 training patch bundle for TRELLIS 1.

What this folder is for
This folder contains the custom patch files needed to add the Architecture-1 training path on top of a fresh upstream TRELLIS checkout.

What this patch adds
- custom dataset loader for preprocessed Arch1 samples
- custom text projection module
- custom trainer wrappers for Stage 1 and Stage 2
- custom conditioning mixin for fused image + text conditioning
- Arch1 smoke and proof-of-concept configs
- patch installer script

Expected workflow
1. Clone upstream microsoft/TRELLIS separately.
2. Copy this arch1_training_patch folder to the machine.
3. Run:
   python install_arch1_training_patch.py
4. This copies the patch files into the TRELLIS repo and updates the needed __init__.py imports.
5. After patching, run the training scripts from the main repo:
   - run_stage1_smoke.sh
   - run_stage2_smoke.sh
   - or the scale scripts for larger runs

Files in this folder
- install_arch1_training_patch.py
- configs/
- trellis/datasets/arch1_editing.py
- trellis/models/arch1_condition.py
- trellis/trainers/flow_matching/arch1_flow_matching.py
- trellis/trainers/flow_matching/mixins/arch1_conditioned.py

Important note
This folder is only the patch bundle.
It is not the full training repo and it is not a full TRELLIS clone.

The full workflow is:
filtered subset -> pseudo-3D assets -> latent targets + conditioning files -> patch fresh TRELLIS -> Stage 1 / Stage 2 training