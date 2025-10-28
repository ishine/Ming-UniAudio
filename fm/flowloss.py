import torch
import torch.nn as nn

from .dit import DiT
from .CFM import CFM


class FlowLoss(nn.Module):
    """Diffusion Loss"""
    def __init__(self, z_channels, llm_cond_dim, **kwargs):
        super(FlowLoss, self).__init__()
        self.z_channels = z_channels
        self.cfm = CFM(
            model=DiT(in_channels=z_channels, llm_cond_dim=llm_cond_dim, **kwargs)
        )

    def forward(self, cond, target, latent_history, mask, patch_size):
        return self.cfm(cond=cond, target=target, latent_history=latent_history, mask=mask, patch_size=patch_size)

    def sample(self, z, latent_history, cfg=1.0, patch_size=1):
        # diffusion loss sampling
        noise = torch.randn(z.shape[0], self.z_channels, latent_history.shape[1]).cuda()
        sampled_token_latent = self.cfm.sample(noise=noise, c=z, latent_history=latent_history, cfg_scale=cfg, patch_size=patch_size)
        return sampled_token_latent
