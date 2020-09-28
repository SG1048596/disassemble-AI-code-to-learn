#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
任务：NER with Bi-LSTM CRF
源码地址：https://pytorch.apachecn.org/docs/1.0/nlp_advanced_tutorial.html
"""
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(1)

# help
def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype = torch.long)

# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    """
    print("vec", vec)
    tensor([[-1.0000e+04, -1.0002e+04, -1.0001e+04, -1.2435e+00, -2.0000e+04]], grad_fn=<AddBackward0>) # vec
    print(argmax(vec))
    3
    print(vec[0, argmax(vec)]) # 取第0,3个数
    tensor(-1.2435, grad_fn=<SelectBackward>) # max_score
    """
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    """
    print(max_score.view(1, -1))
    tensor([[-1.2435]], grad_fn=<ViewBackward>)
    print(vec.size()[1])
    5
    print(max_score.view(1, -1).expand(1, vec.size()[1]))
    tensor([[-1.2435, -1.2435, -1.2435, -1.2435, -1.2435]], grad_fn=<ExpandBackward>) # max_score_broadcast
    print(vec - max_score_broadcast)
    tensor([[ -9999.0000, -10000.3242,  -9999.9922,      0.0000, -19998.8809]], grad_fn=<SubBackward0>)
    print(torch.exp(vec - max_score_broadcast)) # y = e^x
    tensor([[0., 0., 0., 1., 0.]], grad_fn=<ExpBackward>)
    print(torch.sum(torch.exp(vec - max_score_broadcast)))
    tensor(1., grad_fn=<SumBackward0>)
    print(torch.log(torch.sum(torch.exp(vec - max_score_broadcast))))
    tensor(0., grad_fn=<LogBackward>)
    print(max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast))))
    tensor(-1.2435, grad_fn=<AddBackward0>)
    """
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


class BiLSTM_CRF(nn.Module): # 注意继承了nn.Module : __init__  forward
    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True)
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)
        
        # Matrix of transition parameters.  Entry i,j is the score of transitioning *to* i *from* j.
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))
        
        # These two statements enforce the constraint that we never transfer to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000
        
        self.hidden = self.init_hidden()
    
    def init_hidden(self): # 因为是双向LSTM，所以第一个参数是2 (num_layers * num_directions, batch, hidden_size)
        return (torch.randn(2, 1, self.hidden_dim // 2), torch.randn(2, 1, self.hidden_dim // 2))
    
    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        """
        print(init_alphas)
        tensor([[-10000., -10000., -10000., -10000., -10000.]])
        """
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.
        """
        print(init_alphas)
        tensor([[-10000., -10000., -10000.,      0., -10000.]])
        """
        
        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas
        
        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of the previous tag
                emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size)
                """
                print(feat[next_tag])
                tensor(-0.1248, grad_fn=<SelectBackward>)
                print(feat[next_tag].view(1, -1))
                tensor([[-0.1248]], grad_fn=<ViewBackward>)
                print(emit_score)
                tensor([[-0.1248, -0.1248, -0.1248, -0.1248, -0.1248]], grad_fn=<ExpandBackward>)
                """
                # the ith entry of trans_score is the score of transitioning to next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                """
                print(trans_score)
                tensor([[-1.1811e-01, -1.4420e+00, -1.1108e+00, -1.1187e+00, -1.0000e+04]], grad_fn=<ViewBackward>)
                """
                # The ith entry of next_tag_var is the value for the edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                """
                print(next_tag_var)
                tensor([[-1.0000e+04, -1.0002e+04, -1.0001e+04, -1.2435e+00, -2.0000e+04]], grad_fn=<AddBackward0>)
                print(log_sum_exp(next_tag_var))
                tensor(-1.2435, grad_fn=<AddBackward0>)
                print(log_sum_exp(next_tag_var).view(1))
                tensor([-1.2435], grad_fn=<ViewBackward>)
                """
                # The forward variable for this tag is log-sum-exp of all the scores.
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            """
            print(alphas_t)
            [tensor([-1.2435], grad_fn=<ViewBackward>), tensor([1.7122], grad_fn=<ViewBackward>), 
             tensor([0.4553], grad_fn=<ViewBackward>), tensor([-9999.2988], grad_fn=<ViewBackward>), 
             tensor([-0.4335], grad_fn=<ViewBackward>)]
            """
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha
    
    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        """
        print(self.hidden)
        (
            tensor([
                    [[ 0.4701, -0.3090]], # 正向LSTM

                    [[-0.7961,  0.0254]] # 反向LSTM
            ]), 
            tensor([[[ 0.6973, -1.5115]],

                    [[-0.5501,  0.3092]]
            ])
        )
        """
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1) # self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        """
        print(self.word_embeds(sentence))
        tensor([
                [-1.5256, -0.7502, -0.6540, -1.6095, -0.1002],
                [-0.6092, -0.9798, -1.6091, -0.7121,  0.3037],
                [-0.7773, -0.2515, -0.2223,  1.6871,  0.2284],
                [ 0.4676, -0.6970, -1.1608,  0.6995,  0.1991],
                [ 0.8657,  0.2444, -0.6629,  0.8073,  1.1017],
                [-0.1759, -2.2456, -1.4465,  0.0612, -0.6177],
                [-0.7981, -0.1316,  1.8793, -0.0721,  0.1578],
                [-0.7735,  0.1991,  0.0457,  0.1530, -0.4757],
                [-0.1110,  0.2927, -0.1578, -0.0288,  2.3571],
                [-1.0373,  1.5748, -0.6298, -0.9274,  0.5451],
                [ 0.0663, -0.4370,  0.7626,  0.4415,  1.1651]
                ])
        print(embeds)
        tensor([
                [[-1.5256, -0.7502, -0.6540, -1.6095, -0.1002]],

                [[-0.6092, -0.9798, -1.6091, -0.7121,  0.3037]],

                [[-0.7773, -0.2515, -0.2223,  1.6871,  0.2284]],

                [[ 0.4676, -0.6970, -1.1608,  0.6995,  0.1991]],

                [[ 0.8657,  0.2444, -0.6629,  0.8073,  1.1017]],

                [[-0.1759, -2.2456, -1.4465,  0.0612, -0.6177]],

                [[-0.7981, -0.1316,  1.8793, -0.0721,  0.1578]],

                [[-0.7735,  0.1991,  0.0457,  0.1530, -0.4757]],

                [[-0.1110,  0.2927, -0.1578, -0.0288,  2.3571]],

                [[-1.0373,  1.5748, -0.6298, -0.9274,  0.5451]],

                [[ 0.0663, -0.4370,  0.7626,  0.4415,  1.1651]]
                ])
        view : reshape
        """
        lstm_out, self.hidden = self.lstm(embeds, self.hidden) # self.hidden是lstm的h0和C0
        """
        lstm_out, (hn, cn) = self.lstm(embeds, self.hidden)
        print(hn)
        tensor([
                [[-0.0960,  0.0318]],

                [[-0.0835, -0.0227]]
                ])
        print(cn)
        tensor([
                [[-0.1125,  0.1472]],

                [[-0.1460, -0.1395]]
                ])
        """
        """
        print(lstm_out)
        tensor([
                [[-0.0790, -0.1840, -0.0835, -0.0227]],

                [[-0.0755, -0.1615,  0.0527, -0.0230]],

                [[ 0.0047,  0.0684,  0.1928, -0.1368]],

                [[ 0.0239, -0.0227,  0.1221,  0.0355]],

                [[ 0.0887, -0.1895,  0.2595,  0.0327]],

                [[-0.0245, -0.0154, -0.1261, -0.0204]],

                [[-0.3871,  0.0886, -0.3393, -0.1131]],

                [[-0.0845,  0.1228,  0.0417, -0.2081]],

                [[-0.1880, -0.1309,  0.3214,  0.0122]],

                [[ 0.1646, -0.2228,  0.2255, -0.0755]],

                [[-0.0960,  0.0318, -0.3106,  0.0361]]
                ])
        print(self.hidden)
        (tensor([
                [[-0.0960,  0.0318]],

                [[-0.0835, -0.0227]]
                ]), 
         tensor([
                 [[-0.1125,  0.1472]],

                 [[-0.1460, -0.1395]]
                ])
        )
        """
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        """
        print(lstm_out)
        tensor([
                [-0.0790, -0.1840, -0.0835, -0.0227],
                [-0.0755, -0.1615,  0.0527, -0.0230],
                [ 0.0047,  0.0684,  0.1928, -0.1368],
                [ 0.0239, -0.0227,  0.1221,  0.0355],
                [ 0.0887, -0.1895,  0.2595,  0.0327],
                [-0.0245, -0.0154, -0.1261, -0.0204],
                [-0.3871,  0.0886, -0.3393, -0.1131],
                [-0.0845,  0.1228,  0.0417, -0.2081],
                [-0.1880, -0.1309,  0.3214,  0.0122],
                [ 0.1646, -0.2228,  0.2255, -0.0755],
                [-0.0960,  0.0318, -0.3106,  0.0361]
                ])
        """
        lstm_feats = self.hidden2tag(lstm_out)
        """
        print(lstm_feats)
        tensor([
                [-0.2095,  0.1737, -0.3876,  0.4378, -0.3475],# no.0
                [-0.2681,  0.1620, -0.4196,  0.4297, -0.2857],
                [-0.3868,  0.2700, -0.4559,  0.3874, -0.2614],
                [-0.3761,  0.2536, -0.3897,  0.4786, -0.2404],
                [-0.3446,  0.1833, -0.4204,  0.4936, -0.0980],
                [-0.2738,  0.2778, -0.3540,  0.4534, -0.3920],
                [-0.2207,  0.2085, -0.4019,  0.3099, -0.6957],
                [-0.3363,  0.2813, -0.4552,  0.3353, -0.3985],
                [-0.3904,  0.0843, -0.5000,  0.3937, -0.2078],
                [-0.2801,  0.2033, -0.4282,  0.4708, -0.0854],
                [-0.2504,  0.3018, -0.3046,  0.4671, -0.5199] # no.10
                ])
        """
        return lstm_feats

    def _score_sentence(self, feats, tags): # tags是ground_truth tags
        # Gives the score of a provided tag sequence
        score = torch.zeros(1)
        """
        print(score)
        tensor([0.])
        print(tags)
        tensor([0, 1, 2, 2, 2, 2, 0])
        """
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])
        """
        print(tags)
        tensor([3, 0, 1, 2, 2, 2, 2, 0])
        """
        for i, feat in enumerate(feats): # i+1不报错是因为加入了START_TAG
            score = score + self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        """
        print(self.transitions)
        tensor([# B(j)         I            O             START_TAG    STOP_TAG（不会从STOP_TAG传到任何的i）
                [-1.1811e-01, -1.4420e+00, -1.1108e+00, -1.1187e+00, -1.0000e+04], # B(i)
                [-4.9566e-01, -1.9700e-01, -3.3396e-02,  1.4273e+00, -1.0000e+04], # I
                [-7.5307e-01, -4.3190e-01,  6.6930e-01,  6.5051e-01, -1.0000e+04], # O
                [-1.0000e+04, -1.0000e+04, -1.0000e+04, -1.0000e+04, -1.0000e+04], # START_TAG(不会从任何j传到START_TAG的i)
                [ 1.8568e-01, -2.7636e-01, -5.9385e-01, -3.0606e-01, -1.0000e+04]], # STOP_TAG
               requires_grad=True)
        """
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        """
        print(init_vvars)
        tensor([[-10000., -10000., -10000., -10000., -10000.]])
        """
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0
        """
        # tag_to_ix = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4}
        print(init_vvars)
        tensor([[-10000., -10000., -10000.,      0., -10000.]])
        """
        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars

        # feat有两个用处：一个是用于+ self.transitions，用来“别管feat判断的概率最大的tag是什么，
        # 拿上一时刻状态（也是通过上一时刻feat计算得到的）和转移概率来计算：别管从哪个tag转移过来，
        # 总之计算当前输出时点每个tag出现的最大概率”
        # 另一个是 然后用torch.cat(viterbivars_t) + feat来计算新的当前状态，作为下个循环的“上一时刻状态”
        # 为什么新的当前状态不能单独用feat表示呢？可能是为了反向传播？
        # 初始forward_var+transitions=[viterbivars_t]（第1个时刻的各tag最大概率序列list）
        #                                   + feat1
        # START时刻——>               第1个输出的时刻——>      第2个输出的时刻——>
        #                                   =                        + feat2
        #                               forward_var+transitions=[viterbivars_t]
        for feat in feats: # 遍历每个词语对应的输出
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size): # 遍历每个词语对应的5个tag的概率,next_tag = 0,1,2,3,4
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag] # 当前状态+转移到每个标签的概率=实际转移到每个标签的概率
                # 上一时刻可能是5个tag的概率分别为[-10000., -10000., -10000.,      0., -10000.](forward_var)
                # 每个tag转移到tag0（"B"）的概率为forward_var + self.transitions[next_tag]
                """
                print(self.transitions[next_tag]) # 本次循环内self.transitions保持不变
                """
                best_tag_id = argmax(next_tag_var) # 如果要转移到tag0（"B"），则从tag_best_tag_id转移到tag0（"B"）的概率最大
                bptrs_t.append(best_tag_id) # bptrs_t这个list的第0个含义为，转移到tag0（"B"）的最佳前tag是什么
                """
                print("--------------------------------")
                print(next_tag_var)
                print(next_tag_var[0])
                print(best_tag_id)
                print(next_tag_var[0][best_tag_id])
                print(next_tag_var[0][best_tag_id].view(1))
                tensor([[-1.0000e+04, -1.0001e+04, -1.0001e+04, -1.1187e+00, -2.0000e+04]])
                tensor([-1.0000e+04, -1.0001e+04, -1.0001e+04, -1.1187e+00, -2.0000e+04])
                3
                tensor(-1.1187)
                tensor([-1.1187])
                print("--------------------------------")
                """
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1)) # 上述最佳转移对应的概率是多少
            # Now add in the emission scores, and assign forward_var to the set of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1) 
            # feat是该词语对应的词性是B\I\O\START\STOP的概率，
            # torch.cat(viterbivars_t)是上一时刻（从某个tag，取转移到B\I\O\START\STOP的概率最大的tag记录下来）
            # 转移到B\I\O\START\STOP的概率
            backpointers.append(bptrs_t)
            """
            print(backpointers)
            [[3, 3, 3, 3, 3], [1, 1, 1, 1, 1], [1, 1, 2, 1, 1], [1, 1, 2, 1, 1], [1, 1, 2, 1, 1], [2, 1, 2, 1, 1], 
             [2, 2, 2, 2, 1], [2, 2, 2, 2, 1], [2, 2, 2, 1, 1], [2, 2, 2, 2, 1], [2, 2, 2, 2, 1]]
            """
            

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id] # Attention,此处path_score出没

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        """
        print(sentence)
        tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10])
        """
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)
        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq
                

START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 5
HIDDEN_DIM = 4

# Make up some training data
training_data = [(
    "the wall street journal reported today that apple corporation made money".split(),
    "B I I I O O O B I O O".split()
), (
    "georgia tech is a university in georgia".split(),
    "B I O O O O B".split()
)]
"""
print(training_data)
[
    (
        ['the', 'wall', 'street', 'journal', 'reported', 'today', 'that', 'apple', 'corporation', 'made', 'money'], 
        ['B', 'I', 'I', 'I', 'O', 'O', 'O', 'B', 'I', 'O', 'O']
    ), 
    (
        ['georgia', 'tech', 'is', 'a', 'university', 'in', 'georgia'], 
        ['B', 'I', 'O', 'O', 'O', 'O', 'B']
    )
]
"""
word_to_ix = {}
for sentence, tags in training_data:
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
"""           
print(word_to_ix)
{'the': 0, 'wall': 1, 'street': 2, 'journal': 3, 'reported': 4, 
 'today': 5, 'that': 6, 'apple': 7, 'corporation': 8, 'made': 9, 
 'money': 10, 'georgia': 11, 'tech': 12, 'is': 13, 'a': 14, 
 'university': 15, 'in': 16}     
"""
tag_to_ix = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4}

# 参数：词表长度，标签离散化序列，词向量维数，隐藏层维数
model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
"""
print(type(model)) #<class '__main__.BiLSTM_CRF'>         
print(model)
BiLSTM_CRF(
  (word_embeds): Embedding(17, 5)
  (lstm): LSTM(5, 2, bidirectional=True)
  (hidden2tag): Linear(in_features=4, out_features=5, bias=True)
)
"""
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
"""
print(optimizer)
SGD (
Parameter Group 0
    dampening: 0
    lr: 0.01
    momentum: 0
    nesterov: False
    weight_decay: 0.0001
)
"""
# Check predictions before training
with torch.no_grad():
    precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
    """
    print(training_data[0][0])
    ['the', 'wall', 'street', 'journal', 'reported', 'today', 'that', 'apple', 'corporation', 'made', 'money']
    print(precheck_sent)
    tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10])
    """
    precheck_tags = torch.tensor([tag_to_ix[t] for t in training_data[0][1]], dtype=torch.long)
    """
    print(precheck_tags)
    tensor([0, 1, 1, 1, 2, 2, 2, 0, 1, 2, 2])
    """
    print(model(precheck_sent)) # score, tag_seq
    """
    (tensor(2.6907), [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1])
    """

# Make sure prepare_sequence from earlier in the LSTM section is loaded
for epoch in range(1):  # again, normally you would NOT do 300 epochs, it is toy data for sentence, tags in training_data:
        # Step 1\. Remember that Pytorch accumulates gradients. We need to clear them out before each instance
        model.zero_grad()

        # Step 2\. Get our inputs ready for the network, that is, turn them into Tensors of word indices.
        """
        print(sentence)
        ['georgia', 'tech', 'is', 'a', 'university', 'in', 'georgia']
        """
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)

        # Step 3\. Run our forward pass.
        loss = model.neg_log_likelihood(sentence_in, targets)

        # Step 4\. Compute the loss, gradients, and update the parameters by calling optimizer.step()
        loss.backward() # https://pytorch.org/docs/stable/autograd.html
        optimizer.step()

# Check predictions after training
with torch.no_grad():
    precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
    print(model(precheck_sent))
# We got it!