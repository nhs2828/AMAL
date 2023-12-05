
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from textloader import *
from generate import *

#  TODO: 

def maskedCrossEntropy(output: torch.Tensor, target: torch.LongTensor, padcar: int):
    """
    :param output: Tenseur length x batch x output_dim,
    :param target: Tenseur length x batch
    :param padcar: index du caractere de padding
    """
    #  TODO:  Implémenter maskedCrossEntropy sans aucune boucle, la CrossEntropy qui ne prend pas en compte les caractères de padding.
    sm_output = torch.log(torch.softmax(output, dim=-1)) # log softmax
    masque_target = torch.where(target == padcar, 0., 1.)
    index_target = target.unsqueeze(-1) # leng, batch, 1
    loss = -torch.gather(sm_output, dim=-1, index=index_target).squeeze(-1)*masque_target
    return torch.sum(loss)


class RNN(nn.Module):
    def __init__(self, in_dim, out_dim, hidden):
        super().__init__()
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.hidden = hidden
        self.h = None
        self.forecast = False
        self.input2hidden = nn.Linear(in_features=self.in_dim, out_features=self.hidden, bias=False)
        self.hidden2hidden = nn.Linear(in_features=self.hidden, out_features=self.hidden)
        self.hidden2out = nn.Linear(in_features=self.hidden, out_features=self.out_dim)

    def one_step(self, x, h):
        """
            x: input
            h: hidden precedent
        """
        i = self.input2hidden(x)
        h = self.hidden2hidden(h)
        h_suiv = torch.tanh(torch.add(i,h))
        return h_suiv

    def forward(self, x, h):
        len_seq = x.shape[0]
        batch_size = x.shape[1]
        if self.forecast:
            h_full = torch.zeros((len_seq, batch_size, h.shape[1], self.hidden))
        else:
            h_full = torch.zeros((len_seq, batch_size, self.hidden))
        for i in range(len_seq):
            h = self.one_step(x[i], h)
            h_full[i] = h
        return h_full
    
    def decode(self, h):
        out = self.hidden2out(h)
        return out # on va utiliser cross entropy loss donc raw logits, pas besoin de sigmoid ou softmax
    
    def init_h0(self, batch_size, nb_classe = None):
        if nb_classe is not None:
            self.forecast = True
            return torch.zeros((batch_size, nb_classe, self.hidden))
        return torch.zeros((batch_size, self.hidden))


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, vocab_size):
        super().__init__()
        # define les vars
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        # Long term memory
        # Short term memory
        # forget gate
        self.f = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.input_size+self.hidden_size, out_features=self.hidden_size),
            torch.nn.Sigmoid()
        )
        # input gate
        self.i = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.input_size+self.hidden_size, out_features=self.hidden_size),
            torch.nn.Sigmoid()
        )
        self.g = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.input_size+self.hidden_size, out_features=self.hidden_size),
            torch.nn.Tanh()
        )
        # output gate
        self.o = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.input_size+self.hidden_size, out_features=self.hidden_size),
            torch.nn.Sigmoid()
        )
        # decode
        self.decoder = torch.nn.Linear(in_features=self.hidden_size, out_features=self.vocab_size)

    def one_step(self, x, c, h):
        """
        Input:
            x: input au moment t
            c: long term memory au moment t-1
            h: short term memory au moment t-1
        """
        # forget gate
        f_t = self.f(torch.cat([x, h], dim=-1))
        c_t = torch.mul(c, f_t)
        # input gate
        i_t = self.i(torch.cat([x, h], dim=-1))
        g_t = self.g(torch.cat([x, h], dim=-1))
        c_t = torch.add(c_t, torch.mul(i_t, g_t))
        # output gate
        o_t = self.o(torch.cat([x, h], dim=-1))
        h_t = torch.mul(o_t, torch.tanh(c_t))

        return c_t, h_t
    
    def forward(self, x, c, h):
        """
        Input:
            x: input, tensor 3-D (Length, Batch, Embedding dim)
            c: long term memory, tensor 2-D (batch, hidden)
            h: short term memory, tensor 2-D (batch, hidden)
        """
        len_seq, batch_size = x.shape[0], x.shape[1]
        h_full = torch.empty(size=(len_seq, batch_size, self.hidden_size))
        c_full = torch.empty(size=(len_seq, batch_size, self.hidden_size))
        for i in range(len_seq):
            c, h = self.one_step(x[i], c, h)
            h_full[i] = h
            c_full[i] = c
        return c_full, h_full
        

    def init_long_short_term(self, batch_size):
        return torch.zeros(size=(batch_size, self.hidden_size)), torch.zeros(size=(batch_size, self.hidden_size))
    
    def decode(self, h):
        return self.decoder(h)




class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, vocab_size):
        super().__init__()
        # define les vars
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        # Long term memory
        # Short term memory
        # update gate
        self.z = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.input_size+self.hidden_size, out_features=self.hidden_size, bias=False),
            torch.nn.Sigmoid()
        )
        # reset gate
        self.r = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.input_size+self.hidden_size, out_features=self.hidden_size, bias=False),
            torch.nn.Sigmoid()
        )
        # inter
        self.inter = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.input_size+self.hidden_size, out_features=self.hidden_size, bias=False),
            torch.nn.Tanh()
        )
        # decode
        self.decoder = torch.nn.Linear(in_features=self.hidden_size, out_features=self.vocab_size)

    def one_step(self, x, h):
        """
        Input:
            x: input au moment t
            h: memory au moment t-1
        """
        # update gate
        z_t = self.z(torch.cat([x, h], dim=-1))
        # reset gate
        r_t = self.r(torch.cat([x, h], dim=-1))
        # inter -> next hidden
        inter_t = self.inter(torch.cat([torch.mul(r_t, h), x], dim=-1))
        h_t = torch.add(torch.mul((1-z_t),h), torch.mul(z_t, inter_t))
        return h_t # (batch, hidden)
    
    def forward(self, x, h):
        """
        Input:
            x: input, tensor 3-D (Length, Batch, Embedding dim)
            h: memory cell, tensor 2-D (Batch, hidden)
        """
        len_seq, batch_size = x.shape[0], x.shape[1]
        h_full = torch.empty(size=(len_seq, batch_size, self.hidden_size))
        for i in range(len_seq):
            h = self.one_step(x[i], h)
            h_full[i] = h
        return h_full # (len, batch, hidden)
        

    def init_memory(self, batch_size):
        return torch.zeros(size=(batch_size, self.hidden_size))
    
    def decode(self, h):
        return self.decoder(h)

