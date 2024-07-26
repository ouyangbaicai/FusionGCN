import os
import sys

import torch
import numpy as np
from tqdm import tqdm

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, save_path, patience=6, verbose=False, delta=0.):
        """
        Args:
            save_path : 模型保存文件夹
            patience (int): How long to wait after last time validation loss improved.
                            Default: 6
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0.
        """
        self.save_path = save_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, model, val_loss, current_epoch, save_every_model=True):

        torch.save(model.state_dict(), 'debug_model.ckpt')

        if save_every_model:
            model_save_path = self.save_path + '/model' + str(current_epoch) + '.ckpt'
            torch.save(model.state_dict(), model_save_path)

        score = -val_loss

        if self.best_score is None:
            print('')
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'\033[0;33mEarlyStopping counter: {self.counter} out of {self.patience}\033[0m')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            print('')
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            tqdm.write(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # path = os.path.join(self.save_path, 'best_network' + '_ws='  + str(window_size) + '*' + str(window_size) + '_bs=' + str(batch_size) + '_lr=' + str(lr) + '_iav=' +  str(img_argument_val) + '_dim=' + str(dim) + '_depth=' + str(depth) + '_heads=' + str(heads) + '_mlpdim=' + str(mlp_dim) + '.pth')
        path = os.path.join(self.save_path, 'best_network' + '.pth')
        torch.save(model.state_dict(), path)  # 这里会存储迄今最优模型的参数
        self.val_loss_min = val_loss
