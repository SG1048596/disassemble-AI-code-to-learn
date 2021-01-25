"""
http://nlp.seas.harvard.edu/2018/04/03/attention.html
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn
seaborn.set_context(context="talk")

class EncoderDecoder(nn.Module): # 这个类并不在意encoder和decoder的内部实现
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
    
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask) # encoder的作用是生成记忆，给到decoder
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
        # memory = self.encode(src, src_mask)
    
    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                            tgt, tgt_mask)

class Generator(nn.Module):
    # 将decoder的输出映射到词表维数的概率，从而输出最大概率的词语
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)

def clones(module, N):
    # 工具函数
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class LayerNorm(nn.Module):
    # 工具类，用于layer normalization
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6): # feature代表输入的维数
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N) # encoder的每一层的具体构造并不在该类中实现
        self.norm = LayerNorm(layer.size) # encoder网络的最后一层

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
            # print(mask)
            # print(mask.size())
        return self.norm(x)

class SublayerConnection(nn.Module): 
    # 工具类，用于建立residual联系。
    # sublayer是传参进来的某一层网络结构，x是传参进来的某个矩阵输入，
    # 先对输入x做layer normalization，输入则变成了self.norm，
    # 再传参进sublayer网络，再加一层dropout，最后再加上输入x
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask): 
        # 先经过sublayer[0],再经过sublayer[1]
        # sublayer[0]代表了attention的部分，
        # sublayer[1]代表了残差网络的部分
        # sublayer的第一个参数是输入x，第二个参数是输入x要经过的网络函数
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
 
    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    # 返回矩阵的上三角部分，并将第k条对角线下方的元素全部置0
    return torch.from_numpy(subsequent_mask) == 0 # 从01矩阵变成了true false矩阵

"""
An attention function can be described as mapping a query and a set of 
key-value pairs to an output, where the query, keys, values, and output 
are all vectors. The output is computed as a weighted sum of the values, 
where the weight assigned to each value is computed by a compatibility 
function of the query with the corresponding key.
       (similarity)  key  (weight)  value
query  (similarity)  key  (weight)  value
       (similarity)  key  (weight)  value
"""
def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scale Dot Product Attention"
    d_k = query.size(-1) # 命名为d_k是为了和MultiHeadedAttention中的变量名保持一致
    scores = torch.matmul(query, key.transpose(-2, -1)) /math.sqrt(d_k)
    if mask is not None:   # transpose调换数组行列的索引值，正常数组索引值为(0,1,2),等于(x,y,z)。若传参(1,0,2),等于(y,x,z)
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

"""
h=8 parallel attention layers, or heads.
d_k = d_v = 64
"""
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1): 
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h # d_k=64
        self.h = h # h=8, d_model=512词向量维数
        self.linears = clones(nn.Linear(d_model, d_model), 4) # 512*512, 4层，从低维特征到高维特征
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1) # unsqueeze pytorch函数，对数据维数进行扩充
        nbatches = query.size(0) # numpy.size(a, axis=None)用于统计矩阵元素个数，axis=0返回行数，=1返回列数

        # 1) Do all the linear projections in batch from d_model => h x d_k 线性投影，将512维拆分成8个64维
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) 
            for l, x in zip(self.linears, (query, key, value))] 
            # for循环的是结果是：
            # query = self.linears[0](query) 经过一个线性网络后(输出为512维) ——> 8*64(h*d_k)
            # key = self.linears[1](key) 经过一个线性网络后(输出为512维) ——> 8*64(h*d_k)
            # value = self.linears[2](value) 经过一个线性网络后(输出为512维) ——> 8*64(h*d_k)

        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        # 512维切分成8块分别做attention，写到代码里则是以batch的形式

        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k) # d_model=h * d_k
        return self.linears[-1](x)
"""
  1) In “encoder-decoder attention” layers, the queries come from the previous 
  decoder layer, and the memory keys and values come from the output of the 
  encoder. This allows every position in the decoder to attend over all positions 
  in the input sequence. 
  2) The encoder contains self-attention layers. In a self-attention layer all of 
  the keys, values and queries come from the same place, in this case, the output 
  of the previous layer in the encoder. Each position in the encoder can attend to 
  all positions in the previous layer of the encoder.
  3) Similarly, self-attention layers in the decoder allow each position in the 
  decoder to attend to all positions in the decoder up to and including that position. 
  We need to prevent leftward information flow in the decoder to preserve the 
  auto-regressive property. We implement this inside of scaled dot- product attention 
  by masking out (setting to −∞) all values in the input of the softmax which correspond 
  to illegal connections.
""" 

"""
  In addition to attention sub-layers, each of the layers in our encoder and 
  decoder contains a fully connected feed-forward network.
  FFN(x)=max(0,xW1+b1)W2+b2,While the linear transformations are the same 
  across different positions, they use different parameters from layer to 
  layer. Another way of describing this is as two convolutions with kernel 
  size 1. The dimensionality of input and output is d_model=512, and the 
  inner-layer has dimensionality dff=2048.
"""
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

"""
In our model, we share the same weight matrix between the two embedding 
layers and the pre-softmax linear transformation, similar to《Using the 
Output Embedding to Improve Language Models》
"""
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

"""
Add “positional encodings” to the input embeddings at the 
bottoms of the encoder and decoder stacks.The positional 
encodings have the same dimension d_model as the embeddings, 
so that the two can be summed.
pos+k将是pos的线性函数
In addition, we apply dropout to the sums of the embeddings 
and the positional encodings in both the encoder and decoder stacks.
"""
class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)

"""
Here we define a function that takes in hyperparameters and produces a full model.
"""
def make_model(src_vocab, tgt_vocab, N=6, 
               d_model=512, d_ff=2048, h=8, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    # print("src_vocab", src_vocab) 10
    # print("tgt_vocab", tgt_vocab) 10
    # print("N", N) 下面的示例中填了2，6代表encoder中有6层encoderLayer
    c = copy.deepcopy # 简化函数名称
    attn = MultiHeadedAttention(h, d_model) # (8, 512) 1*512
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), 
                             c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab)) # 注意此处的vocab都是维数而非词向量
    # nn.Sequential是一个顺序容器，将特定神经网络模块按照传入构造器的顺序
    # 依次添加到计算图中执行。所以先执行Embeddings再执行position。
    # EncoderLayer是Encoder每一层的具体网络结构，N是一共有几层。
    # 每一层EncoderLayer先是self-attention再是(1,512)*(512,2048)*(2048*512)
    # 且EncoderLayer中的这2层每层都再经由残差连接处理
    # 此处共有N=6层EncoderLayer，即6*2=12
    # 其中每一个self-attention都是先经过一个线性网络后(输出为512维)重塑形状为8*64(h*d_k)后，
    # 走一个attention的(q,k,v)操作后变成有注意力的8*64(h*d_k)后，重塑形状为(1,512)后，
    # 再经过一个线性网络(512*512)输出(1,512)。
    # Decoder不同于Encoder的是每一层DecoderLayer有3层，每层也都再经由残差连接处理
    # 这3层中，先是类似EncoderLayer中的self-attention，
    # 然后再经由(x,m,m)的针对编码器memory的attention操作，
    # 再是(1,512)*(512,2048)*(2048*512)。
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model

# tmp_model = make_model(10, 10, 6)
# print(tmp_model)
"""
EncoderDecoder(
  (encoder): Encoder(
    (layers): ModuleList(
      (0): EncoderLayer(
        (self_attn): MultiHeadedAttention(
          (linears): ModuleList(
            (0): Linear(in_features=512, out_features=512, bias=True)
            (1): Linear(in_features=512, out_features=512, bias=True)
            (2): Linear(in_features=512, out_features=512, bias=True)
            (3): Linear(in_features=512, out_features=512, bias=True)
          )
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=512, out_features=2048, bias=True)
          (w_2): Linear(in_features=2048, out_features=512, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (sublayer): ModuleList(
          (0): SublayerConnection(
            (norm): LayerNorm()
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (1): SublayerConnection(
            (norm): LayerNorm()
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
      )
      (1): EncoderLayer(
        (self_attn): MultiHeadedAttention(
          (linears): ModuleList(
            (0): Linear(in_features=512, out_features=512, bias=True)
            (1): Linear(in_features=512, out_features=512, bias=True)
            (2): Linear(in_features=512, out_features=512, bias=True)
            (3): Linear(in_features=512, out_features=512, bias=True)
          )
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=512, out_features=2048, bias=True)
          (w_2): Linear(in_features=2048, out_features=512, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (sublayer): ModuleList(
          (0): SublayerConnection(
            (norm): LayerNorm()
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (1): SublayerConnection(
            (norm): LayerNorm()
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
      )
    )
    (norm): LayerNorm()
  )
  (decoder): Decoder(
    (layers): ModuleList(
      (0): DecoderLayer(
        (self_attn): MultiHeadedAttention(
          (linears): ModuleList(
            (0): Linear(in_features=512, out_features=512, bias=True)
            (1): Linear(in_features=512, out_features=512, bias=True)
            (2): Linear(in_features=512, out_features=512, bias=True)
            (3): Linear(in_features=512, out_features=512, bias=True)
          )
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (src_attn): MultiHeadedAttention(
          (linears): ModuleList(
            (0): Linear(in_features=512, out_features=512, bias=True)
            (1): Linear(in_features=512, out_features=512, bias=True)
            (2): Linear(in_features=512, out_features=512, bias=True)
            (3): Linear(in_features=512, out_features=512, bias=True)
          )
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=512, out_features=2048, bias=True)
          (w_2): Linear(in_features=2048, out_features=512, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (sublayer): ModuleList(
          (0): SublayerConnection(
            (norm): LayerNorm()
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (1): SublayerConnection(
            (norm): LayerNorm()
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (2): SublayerConnection(
            (norm): LayerNorm()
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
      )
      (1): DecoderLayer(
        (self_attn): MultiHeadedAttention(
          (linears): ModuleList(
            (0): Linear(in_features=512, out_features=512, bias=True)
            (1): Linear(in_features=512, out_features=512, bias=True)
            (2): Linear(in_features=512, out_features=512, bias=True)
            (3): Linear(in_features=512, out_features=512, bias=True)
          )
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (src_attn): MultiHeadedAttention(
          (linears): ModuleList(
            (0): Linear(in_features=512, out_features=512, bias=True)
            (1): Linear(in_features=512, out_features=512, bias=True)
            (2): Linear(in_features=512, out_features=512, bias=True)
            (3): Linear(in_features=512, out_features=512, bias=True)
          )
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=512, out_features=2048, bias=True)
          (w_2): Linear(in_features=2048, out_features=512, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (sublayer): ModuleList(
          (0): SublayerConnection(
            (norm): LayerNorm()
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (1): SublayerConnection(
            (norm): LayerNorm()
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (2): SublayerConnection(
            (norm): LayerNorm()
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
      )
    )
    (norm): LayerNorm()
  )
  (src_embed): Sequential(
    (0): Embeddings(
      (lut): Embedding(10, 512)
    )
    (1): PositionalEncoding(
      (dropout): Dropout(p=0.1, inplace=False)
    )
  )
  (tgt_embed): Sequential(
    (0): Embeddings(
      (lut): Embedding(10, 512)
    )
    (1): PositionalEncoding(
      (dropout): Dropout(p=0.1, inplace=False)
    )
  )
  (generator): Generator(
    (proj): Linear(in_features=512, out_features=10, bias=True)
  )
)
"""
# --------------模型部分结束--------------

"""
Batches and Masking: holds the src and target sentences for training, 
as well as constructing the masks.
"""
class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2) # src_mask是一个true/false矩阵
        if trg is not None:
            self.trg = trg[:, :-1] # 去掉最后一个
            self.trg_y = trg[:, 1:] # 去掉第一个
            self.trg_mask = \
                self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()
        # print("self.src",self.src.size()) # torch.Size([30, 10])
        # print("self.src_mask",self.src_mask.size()) # torch.Size([30, 1, 10])
        # print("self.trg",self.trg.size()) # torch.Size([30, 9])
        # print("self.trg_y",self.trg_y.size()) # torch.Size([30, 9])
        # print("self.trg_mask",self.trg_mask.size()) # torch.Size([30, 9, 9])
        
        # print("y",self.trg_y)
        # print("mask",self.trg_mask)
        # 因为trg_y的每一行，即一句包含9个词的语句，都对应一个9*9的矩阵
        # 该9*9矩阵第一行代表：第1个词语和所有9个词语间的可见关系 
        # [ True, False, False, False, False, False, False, False, False],
        # 该9*9矩阵第二行代表：第2个词语和所有9个词语间的可见关系
        # [ True,  True, False, False, False, False, False, False, False],
        # 以此类推，如下是一个完整的9*9矩阵：
        # [[ True, False, False, False, False, False, False, False, False],
        #  [ True,  True, False, False, False, False, False, False, False],
        #  [ True,  True,  True, False, False, False, False, False, False],
        #  [ True,  True,  True,  True, False, False, False, False, False],
        #  [ True,  True,  True,  True,  True, False, False, False, False],
        #  [ True,  True,  True,  True,  True,  True, False, False, False],
        #  [ True,  True,  True,  True,  True,  True,  True, False, False],
        #  [ True,  True,  True,  True,  True,  True,  True,  True, False],
        #  [ True,  True,  True,  True,  True,  True,  True,  True,  True]]
        # print("self.ntokens",self.ntokens) # tensor(270)
    
    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask

"""
Training Loop: create a generic training and scoring function to keep 
track of loss. We pass in a generic loss compute function that also 
handles parameter updates.
"""
def run_epoch(data_iter, model, loss_compute):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter): # 每生成一个Batch就计算一次loss
        # print("inside",i) # # 0到19，然后再次0到19,和outside同步
        out = model.forward(batch.src, batch.trg, 
                            batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                    (i, loss / batch.ntokens, tokens / elapsed))
            start = time.time()
            tokens = 0
    return total_loss / total_tokens
# --------------通用的训练时的batch和循环输出loss结束--------------

class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup # 热身阶段增加学习率，后续随着step的增加逐渐降低学习率
        self.factor = factor
        self.model_size = model_size # 词向量维数
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate # 改变学习率
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None): # learning-rate的变化函数
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))

# 使用class NoamOpt的示例函数       
# def get_std_opt(model):
#     return NoamOpt(model.src_embed[0].d_model, 2, 4000,
#             torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

"""
During training, we employed label smoothing of value ϵls=0.1(cite). 
This hurts perplexity, as the model learns to be more unsure, but 
improves accuracy and BLEU score.
标签平滑是一种正则化策略，降低真实标签(<1)的权重，抑制过拟合
"""
class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target): # 函数加下划线代表直接在原来的tensor上修改
        assert x.size(1) == self.size
        # print(x, x.size,x.size(0),x.size(1)) # 地址 270 11
        true_dist = x.data.clone() # 后面其实只用到了x的形状
        true_dist.fill_(self.smoothing / (self.size - 2)) 
        # fill_函数的作用是Fills self tensor with the specified value.
        # print(true_dist) # 元素全是0.的矩阵,
        # 如果smoothing=0.4，则true_dist是全为0.4/11-2的矩阵
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        # scatter(dim, index, src) 的参数有3个
        # dim代表沿着哪个维度进行索引，用来scatter的元素索引，
        # 用来scatter的源元素，可以是一个标量或一个张量，
        # 这个scatter可以理解成放置元素或修改元素，
        # 简单来说就是通过一个张量src来修改另一个张量，
        # 哪个元素需要修改、用src中的哪个元素来修改由dim和index决定
        # 如果smoothing=0.4，则confidence=0.6，
        # 即标签为0.6而非1，非标签为0.4而非0
        """
        print(target) # 1*270
        print(true_dist) # 270*11 
tensor([ 6, 10, 10,  1,  5,  1,  4,  3,  1,  7,  1,  6,  9,  5,  7,  2,  4,  5,  3,  6,  4,  2,  2,  1,  9,  6,  4,  7,  9,  7,  4,  2, 10,  1,  6,
         2,  1,  1,  3,  8,  2,  2,  2,  4,  5,  4, 10,  6,  3,  2,  2,  5, 10,  2,  4,  9,  3,  7,  1,  3,  7,  3,  9,  5,  1,  4,  8,  6,  8,  4,
         8,  7,  1,  9,  4,  4,  5, 10,  2,  5,  8,  6,  5,  2,  8,  4,  5,  6,  1,  1,  9,  5,  5,  4,  3,  3,  6,  3,  6, 10,  1,  4,  8,  8,  2,
         10,  5,  8,  8,  3,  4,  8,  4,  7,  3,  5,  6,  1,  3,  1,  2,  2, 10,  5,  2,  2,  9,  1,  6, 10,  2,  3, 10,  7,  6,  3,  9,  8,  7,  5,
         5,  7,  6,  2,  6,  1,  7,  5,  3,  8,  5,  8,  2,  7,  2,  3,  1,  8,  7,  4,  4,  2,  5,  3,  2,  3,  4,  2,  2,  4,  3,  9,  9,  9,  3,
         6,  6,  5,  5,  8,  6, 10,  5,  4,  9,  4, 10,  8,  9,  2,  4, 10,  4,  4,  6,  4,  5,  5, 10,  2,  8,  2,  2,  7,  6,  5, 10,  6,  6, 10,
         9,  7, 10, 10,  2,  2,  8,  5,  4,  4, 10,  4,  7,  7,  8,  7, 10,  7,  2,  8,  6, 10,  9,  8,  5,  6,  1,  6, 10, 10, 10,  7,  9, 10,  6,
         1, 10,  3,  2,  2,  9,  8,  7, 10,  9,  6,  2,  5,  2,  4,  3,  1, 10,  9,  2,  4,  5, 10, 10,  2])
tensor([[0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
        [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
        [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
        [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
        [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
        [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
        [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
        [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
        [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
        [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
        [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
        [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
        [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
        [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
        [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
        [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
        [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
        [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
        [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
        [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
        [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
        [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
        [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
        [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
        [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
        [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
        [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
        [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
        [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
        [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
        [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
        [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
        [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
        [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
        [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
        [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
        [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
        [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
        [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
        [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
        [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
        [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
        [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
        [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
        [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
        [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
        [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
        [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
        [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
        [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
        [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
        [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.]])
"""
        true_dist[:, self.padding_idx] = 0
        # 后面调用该函数时赋值padding_idx=0
        mask = torch.nonzero(target.data == self.padding_idx)
        # torch.nonzero返回的是非零元素的索引组成的矩阵
        # 即，如果target!=0，则target.data == self.padding_idx
        # 为false，即0，nonzero就不返回它的索引。
        # 该语句的含义为：如果target=padding_idx，则mask有该target的索引
        # print(target)
        # tensor([ 3,  7,  9,  6, 10,  7, 10,  6,  4,  6,  3,  8,  7,  3, 10,  4,  9,  9,  1,  9, 10,  7,  8,  5,  1,  2,  7,  5, 10,  5,  1,  3,  1, 10,  4,
        #  6,  5,  8, 10, 10,  6,  7,  8,  8,  5,  5,  8, 10,  7,  4,  8,  4,  9,  4,  5, 10,  9,  9,  2,  3,  6,  6,  9,  6,  6,  7,  7,  2,  9,  8,
        #  6,  6,  6,  2, 10,  3,  3,  9,  1,  2,  3,  6,  8,  2,  6,  3, 10,  9, 10,  8,  7,  5,  1, 10,  7,  2,  6,  4,  4,  2,  5,  3,  7,  6,  6,
        #  3,  1,  7, 10, 10,  5,  2,  1,  6,  2,  8,  4,  5,  6,  4,  9, 10,  2,  9, 10,  7,  4,  1,  4,  7,  5,  9,  8,  1,  6,  7,  6,  7,  2,  5,
        #  6,  7,  9,  4,  1,  2, 10,  2,  7,  7,  8,  6,  4,  2,  1,  9,  4,  2,  6,  1,  3,  2,  5,  2,  1,  1,  3,  5,  9,  2,  7,  3,  3,  9,  8,
        #  7, 10, 10,  9,  9,  7,  1,  7, 10, 10,  9,  2,  3,  6, 10,  2,  9,  6,  9,  9,  2,  8,  6,  9,  9,  8,  6,  3,  5,  9,  9, 10,  9,  6,  7,
        #  1,  6,  7,  6,  5,  7,  1,  9, 10,  6,  1,  3, 10, 10,  7,  9,  9,  6,  5,  2, 10,  5,  8,  2,  9,  7,  8,  6, 10, 10,  3,  7,  8,  9,  5,
        # 10,  1,  4,  9,  6, 10,  9,  1,  9,  4,  6,  8,  3,  7,  3, 10,  4,  1,  7,  5,  6,  8,  9, 10,  4])
        # print(mask)
        # tensor([], size=(0, 1), dtype=torch.int64)
        # print(mask.dim()) # 2
        # print("true_dist",true_dist)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0) # along 0 dim
        # print("true_dist_index_fill",true_dist)
        # 上下两个true_dist相同
        self.true_dist = true_dist
        # print(x)
        return self.criterion(x, Variable(true_dist, requires_grad=False))# compare output to label-smoothing-target

"""
A first example
Given a random set of input symbols from a small vocabulary, 
the goal is to generate back those same symbols.
"""
def data_gen(V, batch, nbatches): 
    # V是不重要的参数，此处代表词向量的数字范围是大于等于1小于V，
    # batch是一个batch里面有多少条数据
    # nbatches是一共有多少个batch
    "Generate random data for a src-tgt copy task."
    for i in range(nbatches):
        # print("outside",i) # 0到19，然后再次0到19
        data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))
        data[:, 0] = 1
        src = Variable(data, requires_grad=False)
        tgt = Variable(data, requires_grad=False)
        # print("src",src.size()) # src torch.Size([30, 10])
        # print("tgt",tgt.size()) # tgt torch.Size([30, 10])
        yield Batch(src, tgt, 0)

"""拆分data_gen yield Batch的过程
def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    print("attn_shape", attn_shape)
    # (1, 9, 9)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    print("subsequent_mask", subsequent_mask)
    # [[[0 1 1 1 1 1 1 1 1]
    #   [0 0 1 1 1 1 1 1 1]
    #   [0 0 0 1 1 1 1 1 1]
    #   [0 0 0 0 1 1 1 1 1]
    #   [0 0 0 0 0 1 1 1 1]
    #   [0 0 0 0 0 0 1 1 1]
    #   [0 0 0 0 0 0 0 1 1]
    #   [0 0 0 0 0 0 0 0 1]
    #   [0 0 0 0 0 0 0 0 0]]]
    print("return", torch.from_numpy(subsequent_mask) == 0)
    # tensor([[[ True, False, False, False, False, False, False, False, False],
    #          [ True,  True, False, False, False, False, False, False, False],
    #          [ True,  True,  True, False, False, False, False, False, False],
    #          [ True,  True,  True,  True, False, False, False, False, False],
    #          [ True,  True,  True,  True,  True, False, False, False, False],
    #          [ True,  True,  True,  True,  True,  True, False, False, False],
    #          [ True,  True,  True,  True,  True,  True,  True, False, False],
    #          [ True,  True,  True,  True,  True,  True,  True,  True, False],
    #          [ True,  True,  True,  True,  True,  True,  True,  True,  True]]])
    return torch.from_numpy(subsequent_mask) == 0

class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, src, trg=None, pad=0):
        print("pad",pad)
        # 0
        self.src = src
        print("self.src",self.src)
        # tensor([[ 1,  1,  4,  4,  8, 10,  4,  6,  3,  5],
        #         [ 1,  7,  9,  9,  2,  7,  8,  8,  9,  2],
        #         [ 1, 10,  9, 10,  5,  4,  1,  4,  6,  1]])
        self.src_mask = (src != pad).unsqueeze(-2) # src_mask是一个true/false矩阵
        print("self.src_mask",self.src_mask)
        # tensor([[[True, True, True, True, True, True, True, True, True, True]],

        #         [[True, True, True, True, True, True, True, True, True, True]],

        #         [[True, True, True, True, True, True, True, True, True, True]]])
        print("trg",trg)
        # tensor([[ 1,  1,  4,  4,  8, 10,  4,  6,  3,  5],
        #         [ 1,  7,  9,  9,  2,  7,  8,  8,  9,  2],
        #         [ 1, 10,  9, 10,  5,  4,  1,  4,  6,  1]])
        if trg is not None:
            self.trg = trg[:, :-1] 
            # 去掉输入进来的trg最后一个，维数变9，所有译码器上一时刻的输入
            print("self.trg",self.trg)
            # tensor([[ 1,  1,  4,  4,  8, 10,  4,  6,  3],
            #         [ 1,  7,  9,  9,  2,  7,  8,  8,  9],
            #         [ 1, 10,  9, 10,  5,  4,  1,  4,  6]])
            self.trg_y = trg[:, 1:] 
            # 去掉输入进来的trg第一个，维数变9，所有译码器下一时刻的输出
            print("self.trg_y",self.trg_y)
            # tensor([[ 1,  4,  4,  8, 10,  4,  6,  3,  5],
            #         [ 7,  9,  9,  2,  7,  8,  8,  9,  2],
            #         [10,  9, 10,  5,  4,  1,  4,  6,  1]])
            self.trg_mask = \
                self.make_std_mask(self.trg, pad)
            print("self.trg_mask",self.trg_mask) 
            # 是9*9的矩阵
            # 每一行，即每一句话的9个词语，对同一句话其他词语的可见程度。
            # tensor([[[ True, False, False, False, False, False, False, False, False],
            #          [ True,  True, False, False, False, False, False, False, False],
            #          [ True,  True,  True, False, False, False, False, False, False],
            #          [ True,  True,  True,  True, False, False, False, False, False],
            #          [ True,  True,  True,  True,  True, False, False, False, False],
            #          [ True,  True,  True,  True,  True,  True, False, False, False],
            #          [ True,  True,  True,  True,  True,  True,  True, False, False],
            #          [ True,  True,  True,  True,  True,  True,  True,  True, False],
            #          [ True,  True,  True,  True,  True,  True,  True,  True,  True]],

            #         [[ True, False, False, False, False, False, False, False, False],
            #          [ True,  True, False, False, False, False, False, False, False],
            #          [ True,  True,  True, False, False, False, False, False, False],
            #          [ True,  True,  True,  True, False, False, False, False, False],
            #          [ True,  True,  True,  True,  True, False, False, False, False],
            #          [ True,  True,  True,  True,  True,  True, False, False, False],
            #          [ True,  True,  True,  True,  True,  True,  True, False, False],
            #          [ True,  True,  True,  True,  True,  True,  True,  True, False],
            #          [ True,  True,  True,  True,  True,  True,  True,  True,  True]],

            #         [[ True, False, False, False, False, False, False, False, False],
            #          [ True,  True, False, False, False, False, False, False, False],
            #          [ True,  True,  True, False, False, False, False, False, False],
            #          [ True,  True,  True,  True, False, False, False, False, False],
            #          [ True,  True,  True,  True,  True, False, False, False, False],
            #          [ True,  True,  True,  True,  True,  True, False, False, False],
            #          [ True,  True,  True,  True,  True,  True,  True, False, False],
            #          [ True,  True,  True,  True,  True,  True,  True,  True, False],
            #          [ True,  True,  True,  True,  True,  True,  True,  True,  True]]])
            self.ntokens = (self.trg_y != pad).data.sum()
            print("self.ntokens",self.ntokens)
            # tensor(27) 3*9
        print("---")
    
    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        print("tgt", tgt) # 9维
        # tensor([[ 1,  1,  4,  4,  8, 10,  4,  6,  3],
        #         [ 1,  7,  9,  9,  2,  7,  8,  8,  9],
        #         [ 1, 10,  9, 10,  5,  4,  1,  4,  6]])
        tgt_mask = (tgt != pad).unsqueeze(-2)
        print("tgt_mask", tgt_mask)
        # tensor([[[True, True, True, True, True, True, True, True, True]],

        #         [[True, True, True, True, True, True, True, True, True]],

        #         [[True, True, True, True, True, True, True, True, True]]])
        print("tgt.size(-1))", tgt.size(-1))
        # 9
        print("tgt_mask.data", tgt_mask.data)
        # tensor([[[True, True, True, True, True, True, True, True, True]],

        #         [[True, True, True, True, True, True, True, True, True]],

        #         [[True, True, True, True, True, True, True, True, True]]])
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        # type_as将张量转换为给定类型的张量
        print("tgt_mask", tgt_mask)
        # tensor([[[ True, False, False, False, False, False, False, False, False],
        #          [ True,  True, False, False, False, False, False, False, False],
        #          [ True,  True,  True, False, False, False, False, False, False],
        #          [ True,  True,  True,  True, False, False, False, False, False],
        #          [ True,  True,  True,  True,  True, False, False, False, False],
        #          [ True,  True,  True,  True,  True,  True, False, False, False],
        #          [ True,  True,  True,  True,  True,  True,  True, False, False],
        #          [ True,  True,  True,  True,  True,  True,  True,  True, False],
        #          [ True,  True,  True,  True,  True,  True,  True,  True,  True]],

        #         [[ True, False, False, False, False, False, False, False, False],
        #          [ True,  True, False, False, False, False, False, False, False],
        #          [ True,  True,  True, False, False, False, False, False, False],
        #          [ True,  True,  True,  True, False, False, False, False, False],
        #          [ True,  True,  True,  True,  True, False, False, False, False],
        #          [ True,  True,  True,  True,  True,  True, False, False, False],
        #          [ True,  True,  True,  True,  True,  True,  True, False, False],
        #          [ True,  True,  True,  True,  True,  True,  True,  True, False],
        #          [ True,  True,  True,  True,  True,  True,  True,  True,  True]],

        #         [[ True, False, False, False, False, False, False, False, False],
        #          [ True,  True, False, False, False, False, False, False, False],
        #          [ True,  True,  True, False, False, False, False, False, False],
        #          [ True,  True,  True,  True, False, False, False, False, False],
        #          [ True,  True,  True,  True,  True, False, False, False, False],
        #          [ True,  True,  True,  True,  True,  True, False, False, False],
        #          [ True,  True,  True,  True,  True,  True,  True, False, False],
        #          [ True,  True,  True,  True,  True,  True,  True,  True, False],
        #          [ True,  True,  True,  True,  True,  True,  True,  True,  True]]])
        return tgt_mask

def data_gen(V, batch, nbatches): 
    # V是不重要的参数，此处代表词向量的数字范围是大于等于1小于V，
    # batch是一个batch里面有多少条数据
    # nbatches是一共有多少个batch
    a = []
    "Generate random data for a src-tgt copy task."
    for i in range(nbatches):
        data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10))) 
        # batch=3(一句话里有几个词语),词向量维数=10，词向量内容为1-10(V=11)
        print(i,'--',data)
        # tensor([[ 6,  1,  4,  4,  8, 10,  4,  6,  3,  5],
        #         [ 8,  7,  9,  9,  2,  7,  8,  8,  9,  2],
        #         [ 6, 10,  9, 10,  5,  4,  1,  4,  6,  1]])
        data[:, 0] = 1
        print(i,'--',data) # 真正的数据，后面都是Variable化
        # tensor([[ 1,  1,  4,  4,  8, 10,  4,  6,  3,  5],
        #         [ 1,  7,  9,  9,  2,  7,  8,  8,  9,  2],
        #         [ 1, 10,  9, 10,  5,  4,  1,  4,  6,  1]])
        src = Variable(data, requires_grad=False)
        print(i,'--',src)
        # tensor([[ 1,  1,  4,  4,  8, 10,  4,  6,  3,  5],
        #         [ 1,  7,  9,  9,  2,  7,  8,  8,  9,  2],
        #         [ 1, 10,  9, 10,  5,  4,  1,  4,  6,  1]])
        tgt = Variable(data, requires_grad=False)
        print(i,'--',tgt)
        # tensor([[ 1,  1,  4,  4,  8, 10,  4,  6,  3,  5],
        #         [ 1,  7,  9,  9,  2,  7,  8,  8,  9,  2],
        #         [ 1, 10,  9, 10,  5,  4,  1,  4,  6,  1]])
        # yield Batch(src, tgt, 0)
        a.append(Batch(src, tgt, 0))

np.random.seed(0)
data_gen(11, 3, 2)
"""

class SimpleLossCompute:
    "A simple loss compute and train function."
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt
        
    def __call__(self, x, y, norm): # __call__使类对象具有类似函数的功能
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)), 
                              y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        # print("===loss.data===", loss.data)
        # print("norm",norm) # tensor(270)
        return loss.data * norm

torch.set_printoptions(threshold=100000, linewidth=150, sci_mode=False)
# 控制台显示控制
np.random.seed(0)
V = 11
criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
model = make_model(V, V, N=2) # 该函数内嵌了512维
# 此时Encoder的forward函数中的mask是torch.Size([30, 1, 10])，且都是True
# model_opt是"Optim wrapper that implements rate."
model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400,
        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
# print(model.src_embed[0].d_model) # 512
# model中的EncoderDecoder有src_embed属性，src_embed又是class Embeddings，有d_model属性
for epoch in range(10):
    model.train() # model是class EncoderDecoder的一个实现，继承了nn.Module
    run_epoch(data_gen(V, 30, 20), model, 
              SimpleLossCompute(model.generator, criterion, model_opt))
    model.eval()
    print(run_epoch(data_gen(V, 30, 5), model,
                    SimpleLossCompute(model.generator, criterion, None)))


