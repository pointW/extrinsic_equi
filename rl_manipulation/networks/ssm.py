from torch import nn
import torch.nn.functional as F
import torch

class SpatialSoftArgmax(nn.Module):
    """Spatial softmax as defined in https://arxiv.org/abs/1504.00702.

    Concretely, the spatial softmax of each feature map is used to compute a weighted
    mean of the pixel locations, effectively performing a soft arg-max over the feature
    dimension.
    """

    def __init__(self, normalize: bool = True) -> None:
        super().__init__()

        self.normalize = normalize

    def _coord_grid(
        self,
        h: int,
        w: int,
        device: torch.device,
    ) -> torch.Tensor:
        if self.normalize:
            return torch.stack(
                torch.meshgrid(
                    torch.linspace(-1, 1, w, device=device),
                    torch.linspace(-1, 1, h, device=device),
                    indexing="ij",
                )
            )
        return torch.stack(
            torch.meshgrid(
                torch.arange(0, w, device=device),
                torch.arange(0, h, device=device),
                indexing="ij",
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4, "Expecting a tensor of shape (B, C, H, W)."

        # Compute a spatial softmax over the input:
        # Given an input of shape (B, C, H, W), reshape it to (B*C, H*W) then apply the
        # softmax operator over the last dimension.
        _, c, h, w = x.shape
        softmax = F.softmax(x.view(-1, h * w), dim=-1)

        # Create a meshgrid of normalized pixel coordinates.
        xc, yc = self._coord_grid(h, w, x.device)

        # Element-wise multiply the x and y coordinates with the softmax, then sum over
        # the h*w dimension. This effectively computes the weighted mean x and y
        # locations.
        x_mean = (softmax * xc.flatten()).sum(dim=1, keepdims=True)
        y_mean = (softmax * yc.flatten()).sum(dim=1, keepdims=True)

        # Concatenate and reshape the result to (B, C*2) where for every feature we have
        # the expected x and y pixel locations.
        return torch.cat([x_mean, y_mean], dim=1).view(-1, c * 2)