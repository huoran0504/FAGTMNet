import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from complexPyTorch.complexLayers import ComplexBatchNorm2d, ComplexConv2d, ComplexLinear
from complexPyTorch.complexFunctions import complex_relu, complex_max_pool2d, complex_matmul, complex_relu
import pandas as pd
from pandas import DataFrame


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.embed_size = 512 #embed_size
        self.hidden_size = 512 #hidden_size
        self.pre_length = configs.pred_len
        self.feature_size = configs.enc_in #channels
        self.seq_length = configs.seq_len
        self.channel_independence = configs.channel_independence
        self.sparsity_threshold = 0.01
        self.scale = 0.02
        self.embeddings = nn.Parameter(torch.randn(1, self.embed_size))
        self.fouriernode = (self.pre_length * self.feature_size) // 2 + 1
        self.Linear1 = ComplexLinear(self.fouriernode, self.fouriernode)
        self.Linear2 = ComplexLinear(self.fouriernode, self.fouriernode)
        self.Linear3 = ComplexLinear(self.fouriernode, self.fouriernode)
        # self.Linear4 = ComplexLinear(self.fouriernode, self.fouriernode)
        # self.Linear5 = ComplexLinear(self.fouriernode, self.fouriernode)

        # self.conv1 = ComplexConv2d(in_channels = 8, out_channels = 8, kernel_size = 7, padding = 3)
        # self.conv2 = ComplexConv2d(in_channels = 8, out_channels = 8, kernel_size = 7, padding = 3)
        # self.conv3 = ComplexConv2d(in_channels = 8, out_channels = 8, kernel_size = 7, padding = 3)

        # self.conv = ComplexConv2d(in_channels = 8, out_channels = 8, kernel_size = 7, padding = 3)


        self.number_frequency = 1
        self.hidden_size_factor = 1
        self.frequency_size = self.embed_size // self.number_frequency
        self.b1 = nn.Parameter(self.scale * torch.randn(2, self.frequency_size * self.hidden_size_factor))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, self.frequency_size))
        self.b3 = nn.Parameter(
            self.scale * torch.randn(2, self.frequency_size * self.hidden_size_factor))


        # self.b4 = nn.Parameter(
        #     self.scale * torch.randn(2, self.frequency_size * self.hidden_size_factor))
        # self.b5 = nn.Parameter(
        #     self.scale * torch.randn(2, self.frequency_size * self.hidden_size_factor))
        self.embeddings_10 = nn.Parameter(torch.randn(self.seq_length, 8))
        self.fc = nn.Sequential(
            nn.Linear(self.embed_size * 8, 512),
            nn.LeakyReLU(),
            nn.Linear(512, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.pre_length)
        )
        self.to('cuda:0')

    def tokenEmb(self, x):
        x = x.unsqueeze(2)
        y = self.embeddings
        return x * y

        # FourierGNN

    def fourierGC(self, x, B, N, L):
        # print(self.args.train_epochs)
        # x.shape = torch.Size([32, 1927, 128])
        # print(x.shape)
        x = x.permute(0, 2, 1)          # torch.Size([8, 512, 49])
        # print(x.shape)
        m1 = self.Linear1(x)
        
        x = x.permute(0, 2, 1)
        m1 = m1.permute(0, 2, 1)

        o1_real = F.relu(
            torch.einsum('bli,bli->bli', x.real, m1.real) - \
            torch.einsum('bli,bli->bli', x.imag, m1.imag) + \
            self.b1[0]
        )

        o1_imag = F.relu(
            torch.einsum('bli,bli->bli', x.imag, m1.imag) + \
            torch.einsum('bli,bli->bli', x.real, m1.real) + \
            self.b1[1]
        )

        # 1 layer
        y = torch.stack([o1_real, o1_imag], dim=-1)
        y = torch.view_as_complex(y)
        y = complex_relu(y)


        o2 = self.Linear2(y.permute(0, 2, 1))
        o2 = o2.permute(0, 2, 1)

        o2_real = F.relu(
            torch.einsum('bli,bli->bli', y.real, o2.real) - \
            torch.einsum('bli,bli->bli', y.imag, o2.imag) + \
            self.b2[0]
        )

        o2_imag = F.relu(
            torch.einsum('bli,bli->bli', y.imag, o2.imag) + \
            torch.einsum('bli,bli->bli', y.real, o2.real) + \
            self.b2[1]
        )

        y = torch.stack([o2_real, o2_imag], dim=-1)
        y = torch.view_as_complex(y)
        y = complex_relu(y)

        o3 = self.Linear3(y.permute(0, 2, 1))
        o3 = o3.permute(0, 2, 1)

        o3_real = F.relu(
            torch.einsum('bli,bli->bli', y.real, o3.real) - \
            torch.einsum('bli,bli->bli', y.imag, o3.imag) + \
            self.b3[0]
        )

        o3_imag = F.relu(
            torch.einsum('bli,bli->bli', y.imag, o3.imag) + \
            torch.einsum('bli,bli->bli', y.real, o3.real) + \
            self.b3[1]
        )

        y = torch.stack([o3_real, o3_imag], dim=-1)
        y = torch.view_as_complex(y)
        y = complex_relu(y)


        # o4 = self.Linear4(y.permute(0, 2, 1))
        # o4 = o4.permute(0, 2, 1)

        # o4_real = F.relu(
        #     torch.einsum('bli,bli->bli', y.real, o4.real) - \
        #     torch.einsum('bli,bli->bli', y.imag, o4.imag) + \
        #     self.b4[0]
        # )

        # o4_imag = F.relu(
        #     torch.einsum('bli,bli->bli', y.imag, o4.imag) + \
        #     torch.einsum('bli,bli->bli', y.real, o4.real) + \
        #     self.b4[1]
        # )

        # y = torch.stack([o4_real, o4_imag], dim=-1)
        # y = torch.view_as_complex(y)
        # y = complex_relu(y)

        # o5 = self.Linear5(y.permute(0, 2, 1))
        # o5 = o5.permute(0, 2, 1)

        # o5_real = F.relu(
        #     torch.einsum('bli,bli->bli', y.real, o5.real) - \
        #     torch.einsum('bli,bli->bli', y.imag, o5.imag) + \
        #     self.b5[0]
        # )

        # o5_imag = F.relu(
        #     torch.einsum('bli,bli->bli', y.imag, o5.imag) + \
        #     torch.einsum('bli,bli->bli', y.real, o5.real) + \
        #     self.b5[1]
        # )

        # y = torch.stack([o5_real, o5_imag], dim=-1)
        # y = torch.view_as_complex(y)
        # y = complex_relu(y)


        return y

    def forward(self, x):
        # print(x.shape)
        x = x.permute(0, 2, 1).contiguous()
        B, N, L = x.shape
        # B*N*L ==> B*NL
        x = x.reshape(B, -1)
        # embedding B*NL ==> B*NL*D
        x = self.tokenEmb(x)

        # FFT B*NL*D ==> B*NT/2*D
        x = torch.fft.rfft(x, dim=1, norm='ortho')
        # print(x.shape)

        x = x.reshape(B, (N * L) // 2 + 1, self.frequency_size)

        bias = x

        # FourierGNN
        x = self.fourierGC(x, B, N, L)

        x = x + bias

        x = x.reshape(B, (N * L) // 2 + 1, self.embed_size)

        # ifft
        x = torch.fft.irfft(x, n=N * L, dim=1, norm="ortho")

        x = x.reshape(B, N, L, self.embed_size)
        x = x.permute(0, 1, 3, 2)  # B, N, D, L

        # projection
        x = torch.matmul(x, self.embeddings_10)
        x = x.reshape(B, N, -1)
        x = self.fc(x)
        x = x.permute(0, 2, 1).contiguous()

        # print(x.shape)

        return x
