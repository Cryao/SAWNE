# -*- coding: utf-8 -*-

import os
import pickle
import random
import argparse
import torch as t
import numpy as np

from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from model import Word2Vec, SGNS
from collections import defaultdict
from sklearn import preprocessing
import datetime
import time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='', help='the vec name')
    parser.add_argument('--alpha', type=float, default=0.5, help='the weight of word vector')
    parser.add_argument('--topk', type=int, default=100, help='the top K word')
    parser.add_argument('--d', type=float, default=0.8, help='the para d ')
    parser.add_argument('--c', type=float, default=0.8, help='the para c')
    parser.add_argument('--sememe', action= 'store_true', help="use sememe kownlege")
    parser.add_argument('--spse', action= 'store_true', help= 'use SPSE sememe vec')
    parser.add_argument('--data_dir', type=str, default='./data/', help="data directory path")
    parser.add_argument('--save_name', type=str, default='_result.txt', help="save file name")
    parser.add_argument('--e_dim', type=int, default=200, help="embedding dimension")
    return parser.parse_args()


class ScorerForSynset:

    def __init__(self, args):
        self.idx2word = pickle.load(open(os.path.join(args.data_dir, 'idx2word.dat'), 'rb'))
        self.idx2sememe = pickle.load(open(os.path.join(args.data_dir, 'idx2sememe.dat'), 'rb'))
        self.widx2sidxs = pickle.load(open(os.path.join(args.data_dir, 'widx2sidxs.dat'), 'rb'))
        self.idx2vec = pickle.load(open(os.path.join(args.data_dir, 'idx2vec.dat'), 'rb'))
        # self.idx2vec = np.load(os.path.join(args.data_dir, 'wordvec.npy'))
        if args.spse and args.sememe:
            self.sidx2vec = pickle.load(open(os.path.join(args.data_dir, 'SPSE_sem2vec.dat'), 'rb'))
        elif args.sememe:
            self.sidx2vec = pickle.load(open(os.path.join("./vec/", args.name + '_sidx2vec.dat'), 'rb'))
        self.word2synset = pickle.load(open(os.path.join(args.data_dir, 'word2synset.dic'), 'rb'))

        self.hyperRelations = pickle.load(open(os.path.join(args.data_dir, 'hyperRelations.dic'), 'rb'))
        self.simRelations = pickle.load(open(os.path.join(args.data_dir, 'simRelations.dic'), 'rb'))

        # self.train_data = pickle.load(open(os.path.join(args.data_dir, 'train.dat'), 'rb'))
        self.test_data = pickle.load(open(os.path.join(args.data_dir, 'test.dat'), 'rb'))

        self.para_nearest_k = args.topk
        self.para_alpha = args.alpha
        self.para_c = args.c
        self.para_d = args.d

        self.args = args

    def predict(self, args):
        # wordvec = np.load(os.path.join(args.data_dir, 'wordvec.npy'))
        # wordvec = t.tensor(wordvec).float()
        vocab_size = len(self.idx2word)
        sememes_size = len(self.idx2sememe)
        print("sememe list has : {}".format(sememes_size))
        # weights = wf if args.weights else None
        save_file = open(os.path.join("./result/", args.name + args.save_name), 'w')
        for iword, owords in self.test_data:
            final, reslist = self.ScorerForSynset(iword)
            # print(iword)
            # print(final)
            word = self.idx2word[iword]
            save_file.write(word + '\n')
            save_file.write(' '.join(final) + '\n')

    def forward(self, widx):
        wvec = self.idx2vec[widx]
        if self.args.sememe:
            sidxs = self.widx2sidxs[widx]
            svecs = self.sidx2vec[sidxs]
            svec = preprocessing.normalize(svecs.sum(0).reshape(1, -1), norm='l2') #------------------------(1)
            # svec = svecs.sum(0).reshape(1, -1)  #-------------------------(2)
            # svec = svecs.mean(0).reshape(1, -1)   #----------------------------(3)
            # print(".....")
            return np.squeeze(self.para_alpha * wvec + (1 - self.para_alpha) * svec)
        else:
            return np.squeeze(wvec)


    def ScorerForSynset(self, target):
        # print(self.idx2word[target])
        vec = self.forward(target)
        res = defaultdict(float)
        nearestwords = []
        for idx, word in enumerate(self.idx2word):
            if (idx==target):
                continue
            wordvec = self.forward(idx)
            # print(vec)
            dotsum = vec.dot(wordvec)
            cosine = dotsum
            nearestwords.append((word,cosine))
        nearestwords.sort(key=lambda x:x[1],reverse=True)
        nearestwords = nearestwords[0:self.para_nearest_k]
        # print(nearestwords)
        rank = 1
        for word, cosine in nearestwords:
            synsets = self.word2synset[word]
            for synset in synsets:
                res[synset]+=cosine*pow(self.para_c, rank)
                if synset in self.hyperRelations:
                    for hyper in self.hyperRelations[synset]:
                        res[hyper]+=cosine*pow(self.para_c, rank) * self.para_d
            rank+=1
        reslist = []
        for synset in res:
            reslist.append((synset, res[synset]))
        reslist.sort(key=lambda x:x[1],reverse=True)
        final = []
        for synset, score in reslist:
            final.append(synset)
        return final,reslist 
    


if __name__ == '__main__':
    oldtime=datetime.datetime.now()

    sc_fun = ScorerForSynset(parse_args())

    sc_fun.predict(parse_args())

    newtime=datetime.datetime.now()
    print(newtime)
    print('Runtime：%s'%(newtime-oldtime))
    print('Runtime：%sms'%(newtime-oldtime).microseconds)
    print('Runtime：%ss'%(newtime-oldtime).seconds)
