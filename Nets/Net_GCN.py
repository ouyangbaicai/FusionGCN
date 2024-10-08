import torch
from torch import nn
from thop import profile, clever_format
import torch.nn.functional as F
from Utilities.CUDA_Check import GPUorCPU
from Nets.GCN_Encoder import SILC_Processes as img_processes
from torch_geometric.nn import GCNConv, BatchNorm, GATConv, LayerNorm
import time

DEVICE = GPUorCPU.DEVICE


class feature_extraction(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(feature_extraction, self).__init__()
        self.at = nn.Mish()
        self.bn = nn.BatchNorm2d(out_dim * 3)
        self.conv5_5 = nn.Conv2d(in_dim, out_dim, kernel_size=5, padding=4, dilation=2)
        self.conv7_7 = nn.Conv2d(in_dim, out_dim, kernel_size=7, padding=3)
        self.conv9_9 = nn.Conv2d(in_dim, out_dim, kernel_size=9, padding=4)

    def forward(self, x):
        conv5 = self.conv5_5(x)
        conv7 = self.conv7_7(x)
        conv9 = self.conv9_9(x)

        out = self.at(self.bn(torch.cat([conv5, conv7, conv9], dim=1)))

        return out


class SimAM(nn.Module):
    def __init__(self, lamda=1e-5):
        super().__init__()
        self.lamda = lamda
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.shape
        n = h * w - 1
        mean = torch.mean(x, dim=[-2, -1], keepdim=True)
        var = torch.sum(torch.pow((x - mean), 2), dim=[-2, -1], keepdim=True) / n
        e_t = torch.pow((x - mean), 2) / (4 * (var + self.lamda)) + 0.5
        out = self.sigmoid(e_t) * x
        return out


class basic_conv(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size1, kernel_size2, scale):
        super(basic_conv, self).__init__()
        self.att = SimAM()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_dim, int(in_dim * scale), kernel_size=kernel_size1, padding=int(kernel_size1 / 2), stride=1),
            nn.BatchNorm2d(int(in_dim * scale)),
            nn.LeakyReLU(),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(int(in_dim * scale), int(in_dim * scale * 2), kernel_size=5, padding=4, stride=1, dilation=2, groups=int(in_dim * scale)),
            nn.BatchNorm2d(int(in_dim * scale*2)),
            nn.ReLU(),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(int(in_dim * scale * 2), in_dim, kernel_size=1, padding=0, stride=1, groups=in_dim),
            nn.Mish(),
        )
        self.block4 = nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size2, padding=int(kernel_size2 / 2), stride=1)

    def forward(self, x):

        residual = x
        x = self.block1(x)
        x = self.block2(x)
        x = self.att(x)
        x = self.block3(x)
        x = x + residual
        out = self.block4(x)

        return out


class conv3x3(nn.Module):
    "3x3 convolution with padding"

    def __init__(self, input_dim, output_dim, stride=1):
        super().__init__()
        self.att = SimAM()
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(output_dim),
            nn.Mish(),
        )

    def forward(self, x):

        x = self.att(x)
        out = self.conv3x3(x)
        return out


class gcn_encoder(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, att_dim, out_dim):
        super(gcn_encoder, self).__init__()
        self.GConv1 = GCNConv(in_dim, hidden_dim)
        self.ln1 = LayerNorm(hidden_dim)
        # self.bn1 = BatchNorm(hidden_dim)
        self.at1 = nn.Mish()
        self.GATConv = GATConv(hidden_dim, att_dim, heads=3)
        self.ln2 = LayerNorm(att_dim * 3)
        # self.bn2 = BatchNorm(att_dim * 3)
        self.at2 = nn.Mish()
        self.GConv2 = GCNConv(att_dim * 3, out_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.GConv1(x, edge_index)
        x = self.ln1(x)
        x = self.at1(x)
        out = x
        x = self.GATConv(x, edge_index)
        x = self.ln2(x)
        x = self.at2(x)
        gcn_result = self.GConv2(x, edge_index)

        return out, gcn_result


class SELayer_2d(nn.Module):
    def __init__(self, channel, reduction):
        super(SELayer_2d, self).__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.linear1 = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.Mish()
        )
        self.linear2 = nn.Sequential(
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, X_input):
        b, c, _, _ = X_input.size()

        y = self.avg_pool(X_input)
        y = y.view(b, c)
        y = self.linear1(y)
        y = self.linear2(y)
        y = y.view(b, c, 1, 1)

        return X_input * y.expand_as(X_input)


class gcn_decoder(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size):
        super(gcn_decoder, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.block = nn.Sequential(
            nn.Conv2d(in_dim, int(in_dim*2), kernel_size=3, padding=1, stride=1, groups=in_dim),
            nn.LeakyReLU(),
            nn.Conv2d(int(in_dim * 2), int(in_dim * 4), kernel_size=5, padding=4, stride=1, groups=int(in_dim*2), dilation=2),
            nn.ReLU(),
        )
        self.se = SELayer_2d(channel=int(in_dim*4), reduction=in_dim)
        self.shuffled_conv = nn.Sequential(
            nn.Conv2d(int(in_dim*4), out_dim, kernel_size=3, padding=1, stride=1, groups=out_dim),
            nn.BatchNorm2d(out_dim),
            nn.Mish()
        )
        # self.out_conv = nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, padding=int(kernel_size / 2), stride=1)


    def forward(self, cnn_result, gcn_result):

        resdual = cnn_result
        x = torch.cat([gcn_result, cnn_result], dim=1)
        x = self.block(x)
        x = self.se(x)
        out = self.shuffled_conv(x)
        # out = self.out_conv(x)
        out = out + resdual
        # modified_channels = x[:, :int(self.in_dim), :, :]
        # unchanged_channels = x[:, :int(self.in_dim), :, :]
        #
        # gcn_result = self.att(gcn_result)
        #
        # modified_channels = modified_channels + gcn_result
        #
        # concatenated = torch.cat((unchanged_channels, modified_channels), dim=1)
        # shuffled_se = self.se(concatenated)
        #
        # out = self.shuffled_conv(shuffled_se)
        # out = self.shuffled_conv(concatenated)

        return out


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.feature_extraction = feature_extraction(in_dim=3, out_dim=6)

        self.gcn_bn1 = nn.BatchNorm2d(18)
        self.gcn_bn2 = nn.BatchNorm2d(36)

        self.mix = nn.Conv2d(6, 1, kernel_size=7, stride=1, padding=3)

        self.cnn_encoder1 = basic_conv(in_dim=18, out_dim=36, kernel_size1=5, kernel_size2=3, scale=2)
        self.cnn_decoder = basic_conv(in_dim=48, out_dim=24, kernel_size1=3, kernel_size2=3, scale=2)

        self.fea_mix = conv3x3(input_dim=72, output_dim=48)
        self.gcn_decoder = gcn_decoder(in_dim=72, out_dim=24, kernel_size=3)

        self.gcn_encoder = gcn_encoder(in_dim=3, hidden_dim=18, att_dim=36, out_dim=36)

        self.out = nn.Sequential(
            nn.Conv2d(24, 1, kernel_size=3, padding=1, stride=1),
            nn.Sigmoid(),
        )

    def forward(self, A, B):
        b, c, h, w = A.shape
        fea_a, fea_b = self.feature_extraction(A), self.feature_extraction(B)
        fea = self.mix(torch.cat([A, B], dim=1))

        gcn_list_a1 = []
        gcn_list_b1 = []
        gcn_list_a2 = []
        gcn_list_b2 = []

        fea_list = torch.split(fea, 1, dim=0)
        fea_list_a = torch.split(A, 1, dim=0)
        fea_list_b = torch.split(B, 1, dim=0)

        for i in range(b):
            gcn_data_a, gcn_data_b, trans_matrix = img_processes(fea_list_a[i], fea_list_b[i], fea_list[i])
            out_a, gcn_result_a = self.gcn_encoder(gcn_data_a)
            gcn_result_a = torch.mm(trans_matrix, gcn_result_a)
            gcn_result_a = gcn_result_a.view(int(h), int(w), 36).permute(-1, 0, 1).unsqueeze(0)
            gcn_list_a1.append(gcn_result_a)
            out_a = torch.mm(trans_matrix, out_a)
            out_a = out_a.view(h, w, 18).permute(-1, 0, 1).unsqueeze(0)
            gcn_list_a2.append(out_a)

            out_b, gcn_result_b = self.gcn_encoder(gcn_data_b)
            gcn_result_b = torch.mm(trans_matrix, gcn_result_b)
            gcn_result_b = gcn_result_b.view(int(h), int(w), 36).permute(-1, 0, 1).unsqueeze(0)
            gcn_list_b1.append(gcn_result_b)
            out_b = torch.mm(trans_matrix, out_b)
            out_b = out_b.view(h, w, 18).permute(-1, 0, 1).unsqueeze(0)
            gcn_list_b2.append(out_b)

        gcn_result_a = self.gcn_bn2(torch.cat(gcn_list_a1, dim=0))
        gcn_result_b = self.gcn_bn2(torch.cat(gcn_list_b1, dim=0))
        gcn_result = self.fea_mix(torch.cat([gcn_result_a, gcn_result_b], dim=1))

        out_a = self.gcn_bn1(torch.cat(gcn_list_a2, dim=0))
        out_b = self.gcn_bn1(torch.cat(gcn_list_b2, dim=0))

        fea_a = fea_a + out_b
        fea_b = fea_b + out_a

        result_a, result_b = self.cnn_encoder1(fea_a), self.cnn_encoder1(fea_b)

        cnn_result = self.fea_mix(torch.cat([result_a, result_b], dim=1))
        cnn_result = self.cnn_decoder(cnn_result)

        # result = result + self.att(gcn_result)

        result = self.gcn_decoder(cnn_result, gcn_result)

        result = self.out(result)

        return result


if __name__ == '__main__':
    test_tensor_A = torch.rand((1, 3, 520, 520)).to(DEVICE)
    test_tensor_B = torch.rand((1, 3, 520, 520)).to(DEVICE)
    model = Network().to(DEVICE)
    model(test_tensor_A, test_tensor_B)
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    print("The number of model parameters: {} M\n".format(round(num_params / 10e5, 6)))
    print(model)
    flops, params = profile(model, inputs=(test_tensor_A, test_tensor_B))
    flops, params = clever_format([flops, params], "%.3f")
    print('flops: {}, params: {}'.format(flops, params))
    result = model(test_tensor_A, test_tensor_B)
    print(result.shape)
