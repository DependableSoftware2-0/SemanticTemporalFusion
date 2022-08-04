import torch

from kornia import image_to_tensor, tensor_to_image
from kornia.augmentation import ColorJitter, RandomChannelShuffle, RandomThinPlateSpline
from kornia.augmentation import RandomVerticalFlip, RandomHorizontalFlip, Resize, RandomCrop, RandomMotionBlur
from kornia.augmentation import RandomEqualize, RandomGaussianBlur, RandomGaussianNoise, RandomSharpness
import kornia as K


class DataAugmentation(torch.nn.Module):
    """Module to perform data augmentation using Kornia on torch tensors."""

    def __init__(self, apply_color_jitter: bool = False) -> None:
        super().__init__()
        self._apply_color_jitter = apply_color_jitter

        self.transforms = nn.Sequential(
            #RandomHorizontalFlip(p=0.50),
            RandomChannelShuffle(p=0.50),
            RandomThinPlateSpline(p=0.50),
            RandomEqualize(p=0.5),
            RandomGaussianBlur((3, 3), (0.1, 2.0), p=0.2),
            RandomGaussianNoise(mean=0., std=1., p=0.2),
            RandomSharpness(0.5, p=0.5)
        )
        self.jitter = ColorJitter(0.5, 0.5, 0.5, 0.5)

    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #x = torch.squeeze(x, dim=1)
        x_out = self.transforms(x)  # BxCxHxW
        if self._apply_color_jitter:
            x_out = self.jitter(x_out)
        return x_out
    
class Preprocess(torch.nn.Module):
    """Module to perform pre-process using Kornia on torch tensors."""
    def __init__(self, img_size: tuple) -> None:
        super().__init__()
        self.crop = RandomCrop(size=img_size, cropping_mode="slice")

    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x) -> torch.Tensor:
        x_tmp: np.ndarray = np.array(x)  # HxWxC
        x_out: Tensor = image_to_tensor(x_tmp, keepdim=True)  # CxHxW
        x_out: Tensor = self.crop(x_out.float()).squeeze(dim=0)
        return x_out.float() / 255.0