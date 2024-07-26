import numpy as np
import torch
from skimage.morphology import remove_small_objects
from Utilities.CUDA_Check import GPUorCPU

DEVICE = GPUorCPU.DEVICE

def Binarization(img_tensor):
    return torch.where(img_tensor > 0.5, 1., 0.)

def RemoveSmallArea(img_tensor, size=None, threshold=0.001):
    if size is None:
        _, _, H, W = img_tensor.shape
        size = threshold * H * W
    img_array = img_tensor.detach().cpu().numpy().astype(np.bool_)
    tmp_image1 = remove_small_objects(img_array, size)
    tmp_image2 = (1 - tmp_image1).astype(np.bool_) # astype(np.bool_)将数组的类型转换为bool
    tmp_image3 = remove_small_objects(tmp_image2, size)
    tmp_image4 = 1 - tmp_image3
    tmp_image4 = tmp_image4.astype(np.float32)

    if type(img_tensor) is torch.Tensor:
        tmp_image4 = torch.from_numpy(tmp_image4)
        tmp_image4 = tmp_image4.to(img_tensor.device)
    return tmp_image4



