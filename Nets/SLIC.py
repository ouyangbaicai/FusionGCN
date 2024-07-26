import torch
import numpy as np
from skimage.segmentation import slic
from Utilities.CUDA_Check import GPUorCPU
from torch_geometric.data import Data
import torch.nn.functional as F
DEVICE = GPUorCPU.DEVICE


def gradient(x):
    _, c, _, _ = x.shape
    kernel = torch.tensor([[[[0, 1, 0],
                            [1, -4, 1],
                            [0, 1, 0]]]], dtype=x.dtype, device=DEVICE, requires_grad=False)
    kernel = kernel.repeat(1, c, 1, 1)
    grad = F.conv2d(x, kernel, padding=1)
    return grad


def sobel(x):
    _, c, _, _ = x.shape
    kernel_x = torch.tensor([[[[-3, 0, 3],
                               [-10, 0, 10],
                               [-3, 0, 3]]]], dtype=x.dtype, device=DEVICE, requires_grad=False)
    kernel_y = torch.tensor([[[[3, 10, 3],
                               [0, 0, 0],
                               [-3, -10, -3]]]], dtype=x.dtype, device=DEVICE, requires_grad=False)
    # edge_x = F.conv2d(x, kernel_x, padding=1)
    # edge_y = F.conv2d(x, kernel_y, padding=1)
    edge_x = torch.zeros_like(x)
    edge_y = torch.zeros_like(x)
    for i in range(c):
        edge_x[:, i, :, :] = F.conv2d(x[:, i, :, :], kernel_x, padding=1)
        edge_y[:, i, :, :] = F.conv2d(x[:, i, :, :], kernel_y, padding=1)
    edge = torch.sqrt(edge_x ** 2 + edge_y ** 2)
    return edge

def SegmentsLabelProcess(labels):
    labels = np.array(labels, np.int64)
    H, W = labels.shape
    ls = list(set(np.reshape(labels, [-1]).tolist()))
    dic = {}
    for i in range(len(ls)):
        dic[ls[i]] = i
    new_labels = labels
    for i in range(H):
        for j in range(W):
            new_labels[i, j] = dic[new_labels[i, j]]

    return new_labels


def SILC_Processes(img_a, img_b, fea_img, scale=0.5):

    _, c, h, w = img_a.shape
    fea_img = fea_img.clone().detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
    init_segments = int(max(h, w) * scale)
    edge_list = []
    edge_img_a = sobel(img_a)
    edge_img_b = sobel(img_b)
    edge_fea_list_a = edge_img_a.clone().detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
    edge_fea_list_b = edge_img_b.clone().detach().cpu().squeeze(0).permute(1, 2, 0).numpy()

    segments = slic(fea_img, n_segments=init_segments, sigma=5)
    if segments.max() + 1 != len(list(set(np.reshape(segments, [-1]).tolist()))): segments = SegmentsLabelProcess(segments)
    superpixel_num = segments.max() + 1  # slic划分出的超像素个数
    segments_flatten = np.reshape(segments, [-1])

    fea_matrix_a = np.zeros([superpixel_num, c], dtype=np.float32)  # 初始化gcn运算特征矩阵 [num, channels]
    fea_matrix_b = np.zeros([superpixel_num, c], dtype=np.float32)  # 初始化gcn运算特征矩阵 [num, channels]
    trans_matrix = torch.zeros([h*w, superpixel_num], dtype=torch.float32).to(DEVICE)  # 转换矩阵，将其特征提取后的转换到源图像上对应 [h*w, num]
    flatten_edge_img_a = np.reshape(edge_fea_list_a, [-1, c])
    flatten_edge_img_b = np.reshape(edge_fea_list_b, [-1, c])

    for i in range(superpixel_num):
        idx = np.where(segments_flatten == i)[0]
        edge_a = flatten_edge_img_a[idx]
        edge_fea_a = np.sum(edge_a, 0)
        edge_b = flatten_edge_img_b[idx]
        edge_fea_b = np.sum(edge_b, 0)

        fea_matrix_a[i] = edge_fea_a
        fea_matrix_b[i] = edge_fea_b
        trans_matrix[idx, i] = 1

    segments_ids = np.unique(segments)
    vs_right = np.vstack([segments[:, :-1].ravel(), segments[:, 1:].ravel()])
    vs_below = np.vstack([segments[:-1, :].ravel(), segments[1:, :].ravel()])
    bneighbors = np.unique(np.hstack([vs_right, vs_below]), axis=1)

    for i in range(bneighbors.shape[1]):
        node1 = bneighbors[0, i]
        node2 = bneighbors[1, i]
        idx1 = np.where(segments_ids == node1)[0][0]
        idx2 = np.where(segments_ids == node2)[0][0]
        edge_list.append(idx1)
        edge_list.append(idx2)

    # Add self loops
    for i in range(len(segments_ids)):
        edge_list.append(i)
        edge_list.append(i)

    fea_matrix_a = torch.from_numpy(fea_matrix_a).to(DEVICE)
    fea_matrix_b = torch.from_numpy(fea_matrix_b).to(DEVICE)

    edge_index = torch.tensor(edge_list).view(-1, 2).to(DEVICE)

    data_a = Data(x=fea_matrix_a, edge_index=edge_index.t())
    data_b = Data(x=fea_matrix_b, edge_index=edge_index.t())
    return data_a, data_b, trans_matrix