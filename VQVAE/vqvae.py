# codes from https://github.com/SingleZombie/DL-Demos/blob/master/dldemos/VQVAE/model.py

from pixcnn import PixelCNN
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(dim, dim, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        tmp = self.relu(x)
        tmp = self.conv1(tmp)
        tmp = self.relu(tmp)
        tmp = self.conv2(tmp)
        return x + tmp


class VQVAE(nn.Module):

    def __init__(self, input_dim, dim, n_embedding):
        """
        :param input_dim: 输入图像的通道数，例如灰度图为1，RGB图为3
        :param dim: codebook中向量e的维度 这里用C表示
        :param n_embedding: codebook中向量的个数 K
        """
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(input_dim, dim, 4, 2, 1),
                                     nn.ReLU(), nn.Conv2d(dim, dim, 4, 2, 1),
                                     nn.ReLU(), nn.Conv2d(dim, dim, 3, 1, 1),
                                     ResidualBlock(dim), ResidualBlock(dim))
        self.vq_embedding = nn.Embedding(n_embedding, dim) ## codebook 维度为KxC
        self.vq_embedding.weight.data.uniform_(-1.0 / n_embedding,
                                               1.0 / n_embedding)
        self.decoder = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            ResidualBlock(dim), ResidualBlock(dim),
            nn.ConvTranspose2d(dim, dim, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(dim, input_dim, 4, 2, 1))
        self.n_downsample = 2

    # def forward(self, x):
    #     # encode
    #     ze = self.encoder(x)
    #
    #     # ze: [N, C, H, W]
    #     # embedding [K, C]
    #     embedding = self.vq_embedding.weight.data
    #     N, C, H, W = ze.shape
    #     K, _ = embedding.shape
    #     embedding_broadcast = embedding.reshape(1, K, C, 1, 1)
    #     ze_broadcast = ze.reshape(N, 1, C, H, W)
    #
    #     distance = torch.sum((embedding_broadcast - ze_broadcast)**2, 2)
    #     """
    #     上面是为了计算每个像素点到codebook中每个向量的距离, 利用了广播机制，embedding_broadcast的shape为[1, K, C, 1, 1]，ze_broadcast的shape为[N, 1, C, H, W]
    #     这样就可以计算每个像素点到每个codebook向量的距离，得到的distance的shape为[N, K, H, W]
    #     具体来说，distance[i, j, h, w]表示第i个z_e的第h行第w列像素点到codebook中第j个向量的距离
    #     """
    #     nearest_neighbor = torch.argmin(distance, 1)
    #     """
    #     nearest_neighbor的shape为[N, H, W]，表示每个像素点对应的codebook向量的索引，注意是索引，不是z_q的值
    #     """
    #     # make C to the second dim
    #     zq = self.vq_embedding(nearest_neighbor).permute(0, 3, 1, 2)
    #     """
    #     self.vq_embedding(nearest_neighbor)输出维度为[N, H, W, C]需要转化为[N, C, H, W] 才能用decoder生成图像
    #     """
    #     # stop gradient
    #     decoder_input = ze + (zq - ze).detach()
    #
    #     # decode
    #     x_hat = self.decoder(decoder_input)
    #     return x_hat, ze, zq

    def forward(self, x, is_train=False):
        # encode
        ze = self.encoder(x)

        # ze: [N, C, H, W]
        # embedding [K, C]
        embedding = self.vq_embedding.weight.data
        N, C, H, W = ze.shape
        K, _ = embedding.shape
        embedding_broadcast = embedding.reshape(1, K, C, 1, 1)
        ze_broadcast = ze.reshape(N, 1, C, H, W)

        distance = torch.sum((embedding_broadcast - ze_broadcast)**2, 2)
        nearest_neighbor = torch.argmin(distance, 1)  # [N, H, W]

        usage_count = torch.zeros(K, device=nearest_neighbor.device) if not is_train else None
        if not is_train:
            # 仅在验证/测试时统计使用频率
            usage_count = self._count_embedding_usage(nearest_neighbor)

        # make C to the second dim
        zq = self.vq_embedding(nearest_neighbor).permute(0, 3, 1, 2)

        # stop gradient
        decoder_input = ze + (zq - ze).detach()

        # decode
        x_hat = self.decoder(decoder_input)

        if is_train:
            return x_hat, ze, zq
        else:
            return x_hat, ze, zq, usage_count

    def _count_embedding_usage(self, nearest_neighbor):
        """
        统计每个嵌入向量的使用情况，并将其归一化成概率
        nearest_neighbor 是形状为 [N, H, W] 的张量，表示每个像素点选择的嵌入向量的索引
        返回一个形状为 [K] 的张量，表示每个嵌入向量的使用概率
        """
        # 获取形状信息
        N, H, W = nearest_neighbor.shape

        # 初始化 usage_count，用来记录每个嵌入向量的使用次数
        usage_count = torch.zeros(self.vq_embedding.num_embeddings, device=nearest_neighbor.device)

        # 使用 scatter_add_ 计算每个嵌入向量的使用次数
        usage_count = usage_count.scatter_add(0, nearest_neighbor.view(-1),
                                              torch.ones(N * H * W, device=nearest_neighbor.device))

        # 计算所有嵌入向量的总使用次数
        total_usage = usage_count.sum()

        # 如果总使用次数大于 0，则进行归一化
        if total_usage > 0:
            usage_count = usage_count / total_usage

        return usage_count

    @torch.no_grad()
    def encode(self, x):
        ze = self.encoder(x)
        embedding = self.vq_embedding.weight.data

        # ze: [N, C, H, W]
        # embedding [K, C]
        N, C, H, W = ze.shape
        K, _ = embedding.shape
        embedding_broadcast = embedding.reshape(1, K, C, 1, 1)
        ze_broadcast = ze.reshape(N, 1, C, H, W)
        distance = torch.sum((embedding_broadcast - ze_broadcast)**2, 2)
        nearest_neighbor = torch.argmin(distance, 1)
        return nearest_neighbor

    @torch.no_grad()
    def decode(self, discrete_latent):
        zq = self.vq_embedding(discrete_latent).permute(0, 3, 1, 2)
        x_hat = self.decoder(zq)
        return x_hat

    # Shape: [C, H, W]
    def get_latent_HW(self, input_shape):
        C, H, W = input_shape
        return (H // 2**self.n_downsample, W // 2**self.n_downsample)