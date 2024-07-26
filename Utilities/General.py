import torch
from torchvision import transforms
from PIL import Image
from Utilities.CUDA_Check import GPUorCPU
import torchvision.transforms as T

DEVICE = GPUorCPU().DEVICE

def Tensor2Img(img_tensor, transform):
    """
    param img_tensor: tensor
    param transforms: torchvision.transforms
    """
    if 'Normalize' in str(transform):
        normal_transform = list(filter(lambda x: isinstance(x, transforms.Normalize), transform.transforms))
        mean = torch.tensor(normal_transform[0].mean, dtype=img_tensor.dtype, device=img_tensor.device)
        std = torch.tensor(normal_transform[0].std, dtype=img_tensor.dtype, device=img_tensor.device)
        img_tensor.mul_(std[:, None, None]).add_(mean[:, None, None])

    img_tensor = img_tensor.transpose(0, 2).transpose(0, 1)  # C x H x W  ---> H x W x C

    if 'ToTensor' in str(transform) or img_tensor.max() < 1:
        img_tensor = img_tensor.detach().cpu().numpy() * 255

    if isinstance(img_tensor, torch.Tensor):
        img_tensor = img_tensor.numpy()

    if img_tensor.shape[2] == 3:
        img = Image.fromarray(img_tensor.astype('uint8')).convert('RGB')
    elif img_tensor.shape[2] == 1:
        img = Image.fromarray(img_tensor.astype('uint8')).squeeze()
    else:
        raise Exception("Invalid img shape, expected 1 or 3 in axis 2, but got {}!".format(img_tensor.shape[2]))

    return img

def denorm(x):
    """Convert the range from [-1, 1] to [0, 1]."""
    out = (x + 1) / 2
    return out.clamp_(0, 1)

def to_same_size(A, focus_map):
    '''
        Input: aim_size_image
               Image_need_to_be_resize
        Output: resized_image
    '''
    A_size = list(A.size())
    focus_map = focus_map.squeeze(dim=0).squeeze(dim=1).cpu()
    focus_map = T.ToPILImage()(focus_map)
    crop_obt = T.Resize((A_size[2], A_size[3]))
    focus_map = crop_obt(focus_map)
    focus_map = T.ToTensor()(focus_map)
    focus_map = focus_map.unsqueeze(dim=0).to(DEVICE)
    return focus_map
