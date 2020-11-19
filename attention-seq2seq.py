"""
任务：Attention
源码地址：https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
"""
from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata # Unicode字符数据库，由一些描述Unicode字符属性和内部关系的纯文本或html文件组成。
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


SOS_token = 0
EOS_token = 1

# 创造词表的类
class Lang:  # 偏语言模型性质
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS
    
    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)
    
    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )
# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")
    
    # Read the file and split into lines
    lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').read().strip().split('\n')
    
    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    """
    print(lines[0])
    Go. Va !
    print(pairs[0])
    ['go .', 'va !']
    """
    
    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs] # reversed(p):<list_reverseiterator object at 0x000001CCBC3E6128>, 所以外面要再套一层list
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)
        
    return input_lang, output_lang, pairs

MAX_LENGTH = 10

# Since there are a lot of example sentences and we want to train something quickly, 
# we’ll trim the data set to only relatively short and simple sentences. 
# Here the maximum length is 10 words (that includes ending punctuation) and  
# we’re filtering to sentences that translate to the form “I am” or “He is” etc. 
# (accounting for apostrophes replaced earlier). 
eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)

def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH and p[1].startswith(eng_prefixes)

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

# The full process for preparing the data is:

# Read text file and split into lines, split lines into pairs
# Normalize text, filter by length and content
# Make word lists from sentences in pairs
def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    """
    print(pairs[0][0]) # va !  reverse=True 法译英
    print(pairs[0][1]) # go .
    """
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs

input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
print(random.choice(pairs))

# seq2seq的优势：the seq2seq model frees us from sequence length and order

# With a seq2seq model the encoder creates a single vector which, in the ideal case, 
# encodes the “meaning” of the input sequence into a single vector — a single point in some N dimensional space of sentences.

class EncoderRNN(nn.Module): # 套路：继承nn.Module
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        
        self.embedding = nn.Embedding(input_size, hidden_size) # input_size是词表大小
        self.gru = nn.GRU(hidden_size, hidden_size) # hidden_size – The number of features in the hidden state h
    
    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden
    
    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden
    
    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        
    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)
        
        attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1) # 输入时刻的词语+上一时刻的隐藏层状态
        # torch.Size([1, 10]) 注意力完全是基于输出语句的上一时刻词语和上一时刻隐藏层状态的，这里的注意力权重输入语句并没有直接参与训练，而是基于输出语句的句子结构来推测对输入语句的（可能会感兴趣的）位置的关注
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0)) # 矩阵乘法
        # torch.Size([1, 1, 256]) = torch.Size([1, 1, 10]) * torch.Size([1, 10, 256]) # 结合了注意力的词向量 = 参数 * 每个时刻从encoder收集的状态
        """
        print(self.attn(torch.cat((embedded[0], hidden[0]), 1)))
        tensor([[ 0.1433,  0.1830,  0.3262, -0.1038, -0.6843,  
                  0.6104,  0.0366,  0.1494, -0.0302, -0.4940]], grad_fn=<AddmmBackward>)
        print(attn_weights)
        tensor([[0.1072, 0.1115, 0.1287, 0.0837, 0.0469, 
                 0.1710, 0.0963, 0.1078, 0.0901, 0.0567]], grad_fn=<SoftmaxBackward>)
        print(attn_weights.unsqueeze(0))
        tensor([[[0.1072, 0.1115, 0.1287, 0.0837, 0.0469, 
                  0.1710, 0.0963, 0.1078, 0.0901, 0.0567]]], grad_fn=<UnsqueezeBackward0>)
        print(attn_weights.size())
        torch.Size([1, 10])
        print(attn_weights.unsqueeze(0).size())
        torch.Size([1, 1, 10]) ##
        print(encoder_outputs)
        tensor([[-0.3139, -0.4735,  0.5000,  ...,  0.1288, -0.0833, -0.1597],
                [-0.5472, -0.3049,  0.4551,  ..., -0.3551, -0.1340, -0.0538],
                [-0.5774, -0.2286, -0.1981,  ..., -0.0189,  0.4777, -0.2943],
                ...,
                [-0.4757, -0.3367,  0.3424,  ..., -0.1544, -0.6825, -0.0865],
                [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
                [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000]], grad_fn=<CopySlices>)
        print(encoder_outputs.size())
        torch.Size([10, 256])
        print(encoder_outputs.unsqueeze(0))
        print(encoder_outputs.unsqueeze(0).size())
        torch.Size([1, 10, 256]) ##
        tensor([[[-0.3139, -0.4735,  0.5000,  ...,  0.1288, -0.0833, -0.1597],
                 [-0.5472, -0.3049,  0.4551,  ..., -0.3551, -0.1340, -0.0538],
                 [-0.5774, -0.2286, -0.1981,  ..., -0.0189,  0.4777, -0.2943],
                 ...,
                 [-0.4757, -0.3367,  0.3424,  ..., -0.1544, -0.6825, -0.0865],
                 [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
                 [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000]]], grad_fn=<UnsqueezeBackward0>)
        print(attn_applied)
        print(attn_applied.size())
        torch.Size([1, 1, 256])
        tensor([[[-0.2252, -0.2228,  0.1951,  0.3941, -0.2334, -0.1872, -0.1414,
                  -0.0411,  0.0048,  0.0652,  0.0145,  0.1532,  0.0357,  0.3088,
                  -0.0462, -0.1718, -0.1033, -0.0782, -0.0908, -0.1895, -0.1365,
                   0.0499,  0.1226,  0.1180,  0.0515,  0.0357,  0.2434, -0.0470,
                   0.1437, -0.0135, -0.1426,  0.1413, -0.2671,  0.0333, -0.1968,
                   0.1410, -0.1548, -0.0829,  0.2343,  0.2228, -0.1768,  0.3084,
                   0.1321,  0.0847,  0.0592,  0.1052,  0.1081, -0.0578, -0.1848,
                   0.0623, -0.2101,  0.0874,  0.0738, -0.1501,  0.1208, -0.0606,
                   0.0844, -0.0461,  0.1918, -0.0788, -0.1096, -0.0633,  0.0900,
                   0.0568, -0.1050,  0.0006,  0.1180, -0.0622,  0.0223, -0.2576,
                  -0.1655,  0.0224,  0.1301, -0.0634, -0.0019, -0.0668,  0.0095,
                  -0.0832, -0.2044,  0.1253, -0.0322, -0.1264, -0.2480,  0.0526,
                   0.0524, -0.0640,  0.0711, -0.0428,  0.1113,  0.1166,  0.0547,
                   0.0555, -0.0161, -0.1241, -0.2312,  0.1833,  0.1280,  0.2451,
                   0.1853, -0.0497, -0.1454,  0.1249,  0.0114, -0.1592, -0.0694,
                  -0.0750, -0.0674, -0.1497, -0.1375, -0.0739, -0.1042, -0.0634,
                  -0.0540, -0.2885, -0.1068, -0.1260,  0.0187, -0.0717, -0.0638,
                   0.1223,  0.0024, -0.0439,  0.0192, -0.1913,  0.0159,  0.0125,
                   0.0293,  0.0192,  0.0607,  0.1785, -0.1929, -0.0870, -0.0904,
                  -0.0652, -0.0882, -0.0871, -0.0986, -0.1199, -0.0291, -0.1004,
                  -0.0281,  0.1030, -0.0486, -0.1478,  0.1485, -0.0626,  0.1256,
                   0.2306,  0.0104,  0.0050,  0.0236, -0.1144, -0.1065, -0.1161,
                  -0.0122,  0.0626, -0.2134,  0.1674,  0.2437, -0.1588,  0.2228,
                   0.0347, -0.1071,  0.0502,  0.0998,  0.1612, -0.1847, -0.1842,
                   0.2253,  0.1001, -0.1038,  0.0734,  0.1363,  0.0587, -0.0539,
                   0.0169,  0.1124,  0.0943, -0.0251, -0.2159, -0.0460, -0.0168,
                   0.0896,  0.1219, -0.0127, -0.1842, -0.0589, -0.3097,  0.1238,
                   0.0075,  0.2554, -0.0751,  0.1109,  0.0979,  0.0972,  0.1241,
                  -0.1145,  0.0465, -0.0611, -0.1552, -0.2089, -0.0489, -0.0604,
                  -0.1117,  0.0331,  0.1830, -0.0046,  0.1877,  0.0487, -0.0478,
                   0.1231, -0.2155, -0.0280, -0.1533,  0.0464, -0.1684,  0.1630,
                  -0.1512, -0.0125,  0.1509,  0.1166, -0.0115,  0.1124,  0.0348,
                  -0.0683, -0.1221,  0.1810, -0.0025, -0.0627, -0.0908, -0.0882,
                  -0.0582, -0.2045, -0.1967,  0.1287, -0.0298, -0.0701,  0.0395,
                  -0.0776, -0.0728, -0.1680,  0.1244,  0.0545,  0.0598,  0.0421,
                  -0.1133,  0.1571, -0.1255,  0.0412,  0.1069, -0.0376,  0.2906,
                  -0.0685, -0.0604, -0.0752, -0.0991]]], grad_fn=<BmmBackward0>)
        """
        
        output = torch.cat((embedded[0], attn_applied[0]), 1) # torch.Size([1, 512]) # cat当前时刻接收的输入词语和注意力
        output = self.attn_combine(output).unsqueeze(0) # torch.Size([1, 1, 256])
        
        output = F.relu(output) # torch.Size([1, 1, 256])
        
        output, hidden = self.gru(output, hidden) # torch.Size([1, 1, 256]) # torch.Size([1, 1, 256])
        
        output = F.log_softmax(self.out(output[0]), dim=1)
        """
        print(output)
        tensor([[-7.9137, -7.8330, -7.8243,  ..., -7.8727, -8.0543, -7.7990]], grad_fn=<LogSoftmaxBackward>)
        print(output.size()) # torch.Size([1, 2803])
        """
        return output, hidden, attn_weights
        
    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)

teacher_forcing_ratio = 0.5

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()
    
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    
    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    
    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
    
    loss = 0
    
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0] # 为attention收集
    
    decoder_input = torch.tensor([[SOS_token]], device=device)
    
    decoder_hidden = encoder_hidden
    
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    
    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length): # encoder_outputs就是前面收集的每个时刻的输出，长度固定为max_length
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing
    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input
            loss += criterion(decoder_output, target_tensor[di]) # loss只加到for循环里面的target_length的长度
            if decoder_input.item() == EOS_token:
                break
    
    loss.backward()
    
    encoder_optimizer.step()
    decoder_optimizer.step()
    
    return loss.item() / target_length
    
import time
import math

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [tensorsFromPair(random.choice(pairs))
                      for i in range(n_iters)]
    criterion = nn.NLLLoss()
    
    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]
        
        loss = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss
        
        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters), iter, iter / n_iters * 100, print_loss_avg))
        
        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
    
    showPlot(plot_losses)


import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
# plt.switch_backend('agg')
# plt.switch_backend('TkAgg')
import matplotlib.ticker as ticker
import numpy as np

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    
def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()
        
        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
        
        for ei in range(input_length):
            encoder_output, encoder_hiddden = encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]
        
        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS
        
        decoder_hidden = encoder_hidden
        
        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)
        
        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])
                
            decoder_input = topi.squeeze().detach()
        
        return decoded_words, decoder_attentions[:di + 1]
    
def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')
        

hidden_size = 256
encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)

trainIters(encoder1, attn_decoder1, 2, print_every=5000) # 75000

evaluateRandomly(encoder1, attn_decoder1)

# You could simply run plt.matshow(attentions) to see attention output displayed as a matrix, 
# with the columns being input steps and rows being output steps
output_words, attentions = evaluate(encoder1, attn_decoder1, "je suis trop froid .")
# plt.matshow(attentions.numpy())

def showAttention(input_sentence, output_words, attentions): # 横坐标是输入语句，从左到右，纵坐标是输出语句，从上到下
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)
    
    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') + ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)
    
    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    
    plt.show()

def evaluateAndShowAttention(input_sentence):
    output_words, attentions = evaluate(encoder1, attn_decoder1, input_sentence)
    print(attentions)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    showAttention(input_sentence, output_words, attentions)

evaluateAndShowAttention("elle a cinq ans de moins que moi .")
# evaluateAndShowAttention("elle est trop petit .")
# evaluateAndShowAttention("je ne crains pas de mourir .")
# evaluateAndShowAttention("c est un jeune directeur plein de talent .")
