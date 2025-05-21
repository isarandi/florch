import torch.nn as nn



class TransitionBatchNorm2d(nn.BatchNorm2d):
    def __init__(
        self,
        num_groups,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
        device=None,
        dtype=None,
    ):
        super().__init__(
            num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
            device=device,
            dtype=dtype,
        )
        self.gn = nn.GroupNorm(num_groups, num_features, eps, affine, device, dtype)
        self.blend: float = 0.0  # To be externally set


    def forward(self, x):
        # if self.blend == 0.0:
        #     return super().forward(x)
        # elif self.blend == 1.0:
        #     return self.gn(x)
        # else:
        #     out_bn = super().forward(x)
        #     out_gn = self.gn(x)
        #     return (1 - self.blend) * out_bn + self.blend * out_gn

        out_bn = super().forward(x)
        out_gn = self.gn(x).to(x.dtype)
        return (1.0 - self.blend) * out_bn + self.blend * out_gn


class GroupNormSameDtype(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x).to(x.dtype)
