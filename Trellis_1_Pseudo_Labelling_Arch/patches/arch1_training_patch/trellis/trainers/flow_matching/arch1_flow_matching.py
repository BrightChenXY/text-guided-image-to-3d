from .mixins.arch1_conditioned import Arch1ConditionedMixin
from .flow_matching import FlowMatchingCFGTrainer
from .sparse_flow_matching import SparseFlowMatchingCFGTrainer


class _Arch1NoSnapshotMixin:
    def snapshot_dataset(self, *args, **kwargs):
        print("Skipping dataset snapshot for Arch1 smoke training.")
        return None

    def snapshot_model(self, *args, **kwargs):
        print("Skipping model snapshot preview for Arch1 smoke training.")
        return None


class Arch1ConditionedFlowMatchingCFGTrainer(
    _Arch1NoSnapshotMixin, Arch1ConditionedMixin, FlowMatchingCFGTrainer
):
    pass


class Arch1ConditionedSparseFlowMatchingCFGTrainer(
    _Arch1NoSnapshotMixin, Arch1ConditionedMixin, SparseFlowMatchingCFGTrainer
):
    pass
