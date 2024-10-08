import os
import sys
import glob
import time
import cv2
import torch
from tqdm import tqdm
from torch import einsum
from Nets.Net_GCN import Network
from Utilities import Consistency
import Utilities.DataLoaderFM as DLr
from torch.utils.data import DataLoader
from Utilities.CUDA_Check import GPUorCPU
DEVICE = GPUorCPU.DEVICE

class ZeroOneNormalize(object):
    def __call__(self, img):
        return img.float().div(255)

class Fusion:
    def __init__(self,
                 modelpath='RunTimeData/best-model.ckpt',
                 dataroot='./Datasets/Eval',    # 如果要跑其他的数据集更改此处文件夹名称
                 dataset_name='Lytro',          # If you want to run another dataset and change the folder name here
                 threshold=0.001,
                 window_size=5,
                 ):
        self.DEVICE = GPUorCPU().DEVICE
        self.MODELPATH = modelpath
        self.DATAROOT = dataroot
        self.DATASET_NAME = dataset_name
        self.THRESHOLD = threshold
        self.window_size = window_size
        self.window = torch.ones([1, 1, self.window_size, self.window_size], dtype=torch.float).to(self.DEVICE)

    def __call__(self, *args, **kwargs):
        if self.DATASET_NAME != None:
            self.SAVEPATH = '/' + self.DATASET_NAME
            self.DATAPATH = self.DATAROOT + '/' + self.DATASET_NAME
            MODEL = self.LoadWeights(self.MODELPATH)
            EVAL_LIST_A, EVAL_LIST_B = self.PrepareData(self.DATAPATH)
            self.FusionProcess(MODEL, EVAL_LIST_A, EVAL_LIST_B, self.SAVEPATH, self.THRESHOLD)
        else:
            print("Test Dataset required!")
            pass

    def LoadWeights(self, modelpath):
        model = Network().to(self.DEVICE)
        model.load_state_dict(torch.load(modelpath))
        model.eval()
        from thop import profile, clever_format
        flops, params = profile(model, inputs=(torch.rand(1, 3, 520, 520).cuda(), torch.rand(1, 3, 520, 520).cuda()))
        flops, params = clever_format([flops, params], "%.5f")
        print('flops: {}, params: {}\n'.format(flops, params))
        return model

    def PrepareData(self, datapath):
        eval_list_A = sorted(glob.glob(os.path.join(datapath, 'sourceA', '*.*')))
        eval_list_B = sorted(glob.glob(os.path.join(datapath, 'sourceB', '*.*')))
        return eval_list_A, eval_list_B

    def ConsisVerif(self, img_tensor, threshold):
        # Verified_img_tensor = Consistency.Binarization(img_tensor)
        # if threshold != 0:
        Verified_img_tensor = Consistency.RemoveSmallArea(img_tensor=img_tensor, threshold=threshold)
        return Verified_img_tensor

    def FusionProcess(self, model, eval_list_A, eval_list_B, savepath, threshold):
        if not os.path.exists('./Results/' + savepath):
            os.makedirs('./Results/' + savepath, exist_ok=True)
        eval_data = DLr.Dataloader_Eval(eval_list_A, eval_list_B)
        eval_loader = DataLoader(dataset=eval_data,
                                 batch_size=1,
                                 shuffle=False, )
        eval_loader_tqdm = tqdm(eval_loader, colour='blue', leave=True, file=sys.stdout)
        cnt = 1
        running_time = []
        with torch.no_grad():
            for A, B in eval_loader_tqdm:
                start_time = time.time()
                D = model(A, B)
                D = torch.where(D > 0.5, 1., 0.)
                D = self.ConsisVerif(D, threshold)
                D = einsum('c w h -> w h c', D[0]).clone().detach().cpu().numpy()
                A = cv2.imread(eval_list_A[cnt - 1])
                B = cv2.imread(eval_list_B[cnt - 1])
                IniF = A * D + B * (1 - D)
                cv2.imwrite('./Results' + savepath + '/' + self.DATASET_NAME + '-' + str(cnt).zfill(2) + '.png', IniF)
                cnt += 1
                running_time.append(time.time() - start_time)
        running_time_total = 0
        for i in range(len(running_time)):
            print("process_time: {} s".format(running_time[i]))
            if i != 0:
                running_time_total += running_time[i]
        print("\navg_process_time: {} s".format(running_time_total / (len(running_time) - 1)))
        print("\nResults are saved in: " + "./Results" + savepath)


if __name__ == '__main__':
    f = Fusion()
    f()
