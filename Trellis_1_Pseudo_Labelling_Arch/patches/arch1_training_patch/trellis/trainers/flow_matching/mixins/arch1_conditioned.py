from typing import *
import torch


class Arch1ConditionedMixin:
    """
    Uses precomputed e_img and e_text from the dataset.
    Fuses them as:
      e_text_proj = text_proj(e_text)
      e_joint = concat([e_img, e_text_proj], dim=1)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def prepare_for_training(**kwargs):
        # No external encoder download needed for training because conditioning is precomputed.
        return

    def _fuse_cond(self, cond: Dict[str, torch.Tensor]) -> torch.Tensor:
        assert isinstance(cond, dict), f'Expected cond dict, got {type(cond)}'
        assert 'img' in cond and 'text' in cond, f'cond keys must be img/text, got {list(cond.keys())}'
        assert 'text_proj' in self.training_models, "Expected model_dict to contain 'text_proj'"

        img = cond['img'].cuda().float()    # [B, T_img, 1024]
        text = cond['text'].cuda().float()  # [B, T_text, 768]

        text_proj = self.training_models['text_proj'](text)  # [B, T_text, 1024]
        joint = torch.cat([img, text_proj], dim=1)           # [B, T_img + T_text, 1024]
        return joint

    def get_cond(self, cond, **kwargs):
        cond = self._fuse_cond(cond)
        kwargs['neg_cond'] = torch.zeros_like(cond)
        cond = super().get_cond(cond, **kwargs)
        return cond

    def get_inference_cond(self, cond, **kwargs):
        cond = self._fuse_cond(cond)
        kwargs['neg_cond'] = torch.zeros_like(cond)
        cond = super().get_inference_cond(cond, **kwargs)
        return cond

    def vis_cond(self, cond, **kwargs):
        # Keep trainer happy if sampling is enabled, but we set i_sample >> max_steps in smoke configs.
        return {}
