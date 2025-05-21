from typing import Optional
import torch.nn.modules.batchnorm
import torch.nn.functional as F
import torch

class _BatchRenorm(torch.nn.modules.batchnorm._NormBase):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: Optional[float] = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        r_min: float = 1 / 3,
        r_max: float = 3.0,
        d_max: float = 5.0,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(
            num_features, eps, momentum, affine, track_running_stats, **factory_kwargs
        )
        self.r_min = r_min
        self.r_max = r_max
        self.d_max = d_max

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked.add_(1)  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """

        # If buffers are not to be tracked, ensure that they won't be updated
        running_mean = self.running_mean if not self.training or self.track_running_stats else None
        running_var = self.running_var if not self.training or self.track_running_stats else None

        if bn_training:
            dtype = input.dtype
            norm_dims = list(range(input.ndim))
            norm_dims.remove(1)
            var, mean = torch.var_mean(input, dim=norm_dims)
            mean = mean.detach()
            var = var.detach()
            moving_var_plus_eps = (self.running_var + self.eps).to(dtype)
            r = torch.sqrt((var + self.eps).to(dtype) / moving_var_plus_eps)
            d = (mean - self.running_mean).to(dtype) * torch.rsqrt(moving_var_plus_eps)
            r = torch.clamp(r, self.r_min, self.r_max).to(dtype)
            d = torch.clamp(d, -self.d_max, self.d_max).to(dtype)

            num_dims_after_c = input.ndim - 2
            r_ = r.reshape(r.shape + (1,) * num_dims_after_c)
            d_ = d.reshape(d.shape + (1,) * num_dims_after_c)
            w_ = self.weight.reshape(self.weight.shape + (1,) * num_dims_after_c).to(dtype)
            b_ = self.bias.reshape(self.bias.shape + (1,) * num_dims_after_c).to(dtype)
            bn_output = F.batch_norm(
                input,
                running_mean,
                running_var,
                self.weight,
                torch.zeros_like(self.bias),
                bn_training,
                exponential_average_factor,
                self.eps,
            )
            out_dtype = bn_output.dtype
            return bn_output * r_.to(out_dtype) + (d_ * w_ + b_).to(out_dtype)
        else:
            return F.batch_norm(
                input,
                running_mean,
                running_var,
                self.weight,
                self.bias,
                bn_training,
                exponential_average_factor,
                self.eps,
            )


class BatchRenorm1d(_BatchRenorm):
    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError(f"expected 2D or 3D input (got {input.dim()}D input)")

class BatchRenorm2d(_BatchRenorm):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError(f"expected 4D input (got {input.dim()}D input)")

class BatchRenorm3d(_BatchRenorm):
    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError(f"expected 5D input (got {input.dim()}D input)")