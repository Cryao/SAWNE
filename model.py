# -*- coding: utf-8 -*-

import numpy as np
import torch as t
import torch.nn as nn

from torch import LongTensor as LT
from torch import FloatTensor as FT


class Bundler(nn.Module):

    def forward(self, data):
        raise NotImplementedError

    def forward_i(self, data):
        raise NotImplementedError

    def forward_o(self, data):
        raise NotImplementedError


class Word2Vec(Bundler):

    def __init__(self, wordvec= None, widx2sidxs = None, vocab_size=20000, sememes_size=2000, embedding_size=200, padding_idx=0, alpha=0.5):
        super(Word2Vec, self).__init__()
        self.alpha = alpha
        self.widx2sidxs = LT(widx2sidxs)
        self.vocab_size = vocab_size
        self.sememes_size = sememes_size
        self.embedding_size = embedding_size
        self.ivectors = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=padding_idx)
        self.ivectors.weight = nn.Parameter(FT(wordvec))
        # if wordvec is not None:
        #     self.ivectors.weight = nn.Parameter(FT(wordvec))
        # else:
        #     self.ivectors.weight = nn.Parameter(t.cat([t.zeros(1, self.embedding_size), FT(self.vocab_size - 1, self.embedding_size).uniform_(-0.5 / self.embedding_size, 0.5 / self.embedding_size)]))
        # self.ovectors = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=padding_idx)
        self.svectors = nn.Embedding(self.sememes_size, self.embedding_size, padding_idx=padding_idx)
        self.svectors.weight = nn.Parameter(FT(self.sememes_size, self.embedding_size).uniform_(-0.5 / self.embedding_size, 0.5 / self.embedding_size))
        # self.ivectors.weight = nn.Parameter(t.cat([t.zeros(1, self.embedding_size), FT(self.vocab_size - 1, self.embedding_size).uniform_(-0.5 / self.embedding_size, 0.5 / self.embedding_size)]))
        # self.ovectors.weight = nn.Parameter(t.cat([t.zeros(1, self.embedding_size), FT(self.vocab_size - 1, self.embedding_size).uniform_(-0.5 / self.embedding_size, 0.5 / self.embedding_size)]))
        # self.ovectors.weight.requires_grad = False
        self.ivectors.weight.requires_grad = False
        self.svectors.weight.requires_grad = True

    def forward(self, data):
        return self.forward_i(data)

    def forward_i(self, data):
        v = LT(data)
        s = self.widx2sidxs[v]
        v = v.cuda() if self.ivectors.weight.is_cuda else v
        s = s.cuda() if self.svectors.weight.is_cuda else s
        # w = v[:, 0]
        # s = v[:, 1:]
        wordvec = self.ivectors(v)
        sememevec = nn.functional.normalize(self.svectors(s).sum(1), dim= 1, p = 2) #-------------------(1)
        # sememevec = self.svectors(s).sum(1) #--------------------(2)
        # sememevec = self.svectors(s).mean(1) #---------------------(3)
        return self.alpha*wordvec + (1 - self.alpha)*sememevec

    def forward_o(self, data):
        v = LT(data)
        s = self.widx2sidxs[v]
        v = v.cuda() if self.ivectors.weight.is_cuda else v
        s = s.cuda() if self.svectors.weight.is_cuda else s
        # print(v.size())
        # w = v[:, :, 0]
        # s = v[:, :, 1:]
        wordvec = self.ivectors(v)
        # print(s.size())
        sememevec = nn.functional.normalize(self.svectors(s).sum(2), dim = 1, p = 2) #-------------------(1)
        # sememevec = self.svectors(s).sum(2) #----------------(2)
        # sememevec = self.svectors(s).mean(2) #----------------------(3)
        # print(wordvec.size(), sememevec.size())
        return self.alpha*wordvec + (1-self.alpha)*sememevec


class SGNS(nn.Module):

    def __init__(self, embedding, vocab_size=20000, n_negs=20, weights=None):
        super(SGNS, self).__init__()
        self.embedding = embedding
        self.vocab_size = vocab_size
        self.n_negs = n_negs
        # self.weights = None #????
        self.weights = weights
        if weights is not None:
            wf = np.power(weights, 0.75)
            wf = wf / wf.sum()
            self.weights = FT(wf)

    def forward(self, iword, owords):
        batch_size = iword.size()[0]
        context_size = owords.size()[1]
        if self.weights is not None:
            nwords = t.multinomial(self.weights, batch_size * context_size * self.n_negs, replacement=True).view(batch_size, -1)
        else:
            nwords = FT(batch_size, context_size * self.n_negs).uniform_(0, self.vocab_size - 1).long()
        ivectors = self.embedding.forward_i(iword).unsqueeze(2)
        ovectors = self.embedding.forward_o(owords)
        nvectors = self.embedding.forward_o(nwords).neg()
        # oloss = t.clamp(t.bmm(ovectors, ivectors).squeeze().sigmoid().log().mean(1), 1e-4, 1)
        # nloss = t.clamp(t.bmm(nvectors, ivectors).squeeze().sigmoid().log().view(-1, context_size, self.n_negs).sum(2).mean(1), 1e-4, 1)
        oloss = t.bmm(ovectors, ivectors).squeeze().sigmoid().log().mean(1)
        nloss = t.bmm(nvectors, ivectors).squeeze().sigmoid().log().view(-1, context_size, self.n_negs).sum(2).mean(1)
        # print(oloss)
        # print(nloss)
        return -(oloss + nloss).mean()#.sum()#.mean()
