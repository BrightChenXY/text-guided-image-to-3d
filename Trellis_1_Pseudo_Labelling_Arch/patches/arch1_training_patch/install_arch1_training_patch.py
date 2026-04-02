import shutil
from pathlib import Path

PATCH_ROOT = Path(__file__).resolve().parent
TRELLIS_ROOT = Path('/workspace/TRELLIS')

copy_pairs = [
    (PATCH_ROOT / 'trellis' / 'models' / 'arch1_condition.py',
     TRELLIS_ROOT / 'trellis' / 'models' / 'arch1_condition.py'),
    (PATCH_ROOT / 'trellis' / 'datasets' / 'arch1_editing.py',
     TRELLIS_ROOT / 'trellis' / 'datasets' / 'arch1_editing.py'),
    (PATCH_ROOT / 'trellis' / 'trainers' / 'flow_matching' / 'mixins' / 'arch1_conditioned.py',
     TRELLIS_ROOT / 'trellis' / 'trainers' / 'flow_matching' / 'mixins' / 'arch1_conditioned.py'),
    (PATCH_ROOT / 'trellis' / 'trainers' / 'flow_matching' / 'arch1_flow_matching.py',
     TRELLIS_ROOT / 'trellis' / 'trainers' / 'flow_matching' / 'arch1_flow_matching.py'),
    (PATCH_ROOT / 'configs' / 'arch1_stage1_smoke.json',
     TRELLIS_ROOT / 'configs' / 'arch1_stage1_smoke.json'),
    (PATCH_ROOT / 'configs' / 'arch1_stage2_smoke.json',
     TRELLIS_ROOT / 'configs' / 'arch1_stage2_smoke.json'),
]

for src, dst in copy_pairs:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f'Copied {src} -> {dst}')

def append_if_missing(path: Path, text: str):
    content = path.read_text()
    if text.strip() in content:
        print(f'Already patched: {path}')
        return
    if not content.endswith('\n'):
        content += '\n'
    content += '\n' + text + '\n'
    path.write_text(content)
    print(f'Patched {path}')

append_if_missing(
    TRELLIS_ROOT / 'trellis' / 'models' / '__init__.py',
    "from .arch1_condition import TextImageConditionProjector"
)

append_if_missing(
    TRELLIS_ROOT / 'trellis' / 'datasets' / '__init__.py',
    "from .arch1_editing import Arch1ConditionedSparseStructureLatent, Arch1ConditionedSLat"
)

append_if_missing(
    TRELLIS_ROOT / 'trellis' / 'trainers' / '__init__.py',
    "from .flow_matching.arch1_flow_matching import Arch1ConditionedFlowMatchingCFGTrainer, Arch1ConditionedSparseFlowMatchingCFGTrainer"
)

print('\\nDone.')
