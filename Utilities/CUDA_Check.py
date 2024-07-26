import torch

class GPUorCPU:
    if torch.cuda.is_available():
        DEVICE = "cuda"
        # print('\nCUDA is available. Calculation is performing on ' + str(
        #     torch.cuda.get_device_name(torch.cuda.current_device())) + '.\n')
    else:
        DEVICE = 'cpu'
        # print('\nOOPS! CUDA is not available! Calculation is performing on CPU.\n')
