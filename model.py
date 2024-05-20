from __future__ import division
import torch
from torch.autograd import Variable
import torch.nn as nn
from utils import outputActivation
import torch.nn.functional as F
from math import sqrt


def positional_encoding(X, num_features, dropout_p=0.1, max_len=512):
    # r'''
    #     给输入加入位置编码
    # 参数：
    #     - num_features: 输入进来的维度
    #     - dropout_p: dropout的概率，当其为非零时执行dropout
    #     - max_len: 句子的最大长度，默认512
    #
    # 形状：
    #     - 输入： [batch_size, seq_length, num_features]
    #     - 输出： [batch_size, seq_length, num_features]
    #
    # 例子：
    #     >>> X = torch.randn((2,4,10))
    #     >>> X = positional_encoding(X, 10)
    #     >>> print(X.shape)
    #     >>> torch.Size([2, 4, 10])
    # '''

    dropout = nn.Dropout(dropout_p)
    P = torch.zeros((1, max_len, num_features))
    X_ = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(
        10000,
        torch.arange(0, num_features, 2, dtype=torch.float32) / num_features)
    P[:, :, 0::2] = torch.sin(X_)
    P[:, :, 1::2] = torch.cos(X_)
    X = X + P[:, :X.shape[1], :].cuda()
    return dropout(X)


class highwayNet(nn.Module):

    ## Initialization
    def __init__(self, args):
        super(highwayNet, self).__init__()

        self.transformer = nn.Transformer(args['d_model'], args['n_head'], args['num_encoder_layers'],
                                          args['num_decoder_layers'])

        self.em_64_2 = nn.Linear(64, 2)
        self.em_39_5 = nn.Linear(39, 5)

        self.hist_emb = torch.nn.Linear(2, args['d_model'])

        self.nbrs_emb = torch.nn.Linear(2, 64)
        ## Unpack arguments
        self.args = args

        ## Use gpu flag
        self.use_cuda = args['use_cuda']

        # Flag for train mode (True) vs test-mode (False)
        self.train_flag = args['train_flag']

        ## Sizes of network layers
        self.encoder_size = args['encoder_size']
        self.decoder_size = args['decoder_size']
        self.in_length = args['in_length']
        self.out_length = args['out_length']
        self.grid_size = args['grid_size']

        # self-multi-parameters
        self.dim_in = args['dim_in']
        self.dim_k = args['dim_k']
        self.dim_v = args['dim_v']
        self.num_heads = args['num_heads']
        self.norm_fact = 1 / sqrt(self.dim_k // self.num_heads)

        self.input_embedding_size = args['input_embedding_size']

        ## Define network weights

        # Input embedding layer
        self.ip_emb = torch.nn.Linear(2, self.input_embedding_size)
        self.em_32_64 = torch.nn.Linear(32, 64)

        self.em_1024_64 = torch.nn.Linear(1024, 64)

        # self
        self.position_emb = nn.Parameter(torch.rand((128, 3, 13)))
        self.graph_fl = torch.nn.Linear(39, 16)
        self.graph_fl2 = torch.nn.Linear(1, 32)

        # Encoder LSTM
        self.enc_lstm1 = torch.nn.LSTM(self.input_embedding_size, self.encoder_size, 1)

        # Encoder LSTM
        self.enc_lstm2 = torch.nn.LSTM(self.input_embedding_size, self.encoder_size, 1)

        self.spatial_embedding = nn.Linear(5, self.encoder_size)

        self.tanh = nn.Tanh()

        self.pre4att = nn.Sequential(
            nn.Linear(self.encoder_size, 1),
        )

        self.dec_lstm = torch.nn.LSTM(self.encoder_size, self.decoder_size)

        # Output layers:
        self.op = torch.nn.Linear(self.decoder_size, 2)  # 2-dimension (x, y)

        # Activations:
        self.leaky_relu = torch.nn.LeakyReLU(0.1)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)

        # self-multi-function-define
        self.linear_q = nn.Linear(self.dim_in, self.dim_k, bias=False)
        self.linear_k = nn.Linear(self.dim_in, self.dim_k, bias=False)
        self.linear_v = nn.Linear(self.dim_in, self.dim_v, bias=False)
        self.pre2att = nn.Sequential(nn.Linear(16, 1))

    def attention(self, lstm_out_weight, lstm_out):
        alpha = F.softmax(lstm_out_weight, 1)

        lstm_out = lstm_out.permute(0, 2, 1)

        new_hidden_state = torch.bmm(lstm_out, alpha).squeeze(2)
        new_hidden_state = F.relu(new_hidden_state)

        return new_hidden_state, alpha

    ## Forward Pass
    def forward(self, hist, nbrs, masks, lat_enc, lon_enc):

        hist_tem = self.leaky_relu(self.hist_emb(hist))
        hist_tem = hist_tem.permute(1, 0, 2)
        # hist_tem_pe = hist_tem.permute(1, 0, 2)
        hist_tem_pe = positional_encoding(hist_tem, 64).permute(1, 0, 2)

        nbrs_out, (nbrs_enc, _) = self.enc_lstm1(self.leaky_relu(self.ip_emb(nbrs)))

        nbrs_out = nbrs_out.permute(1, 0, 2)  # (16, 927, 64)
        nbrs_enc = self.leaky_relu(self.em_1024_64(nbrs_out.flatten(1)))

        soc_enc = torch.zeros_like(masks).float()  # mask size: (128, 3, 13, 64)
        soc_enc = soc_enc.masked_scatter_(masks, nbrs_enc).permute(0, 3, 2, 1).flatten(2).permute(0, 2, 1)
        # soc_enc_pe = positional_encoding(soc_enc, 64).permute(1, 0, 2)
        soc_enc_pe = soc_enc.permute(1, 0, 2)

        out = self.transformer(hist_tem_pe, soc_enc_pe)
        out = self.leaky_relu(self.em_64_2(out).permute(2, 1, 0))
        out = outputActivation(self.em_39_5(out).permute(2, 1, 0))

        return out

    def decode(self, enc):
        enc = enc.repeat(self.out_length, 1, 1)

        h_dec, _ = self.dec_lstm(enc)
        h_dec = h_dec.permute(1, 0, 2)
        fut_pred = self.op(h_dec)
        fut_pred = fut_pred.permute(1, 0, 2)
        fut_pred = outputActivation(fut_pred)
        return fut_pred
