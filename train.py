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

import datetime
import time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='sgns', help="model name")
    parser.add_argument('--data_dir', type=str, default='./data/', help="data directory path")
    parser.add_argument('--save_dir', type=str, default='./pts/', help="model directory path")
    parser.add_argument('--e_dim', type=int, default=200, help="embedding dimension")
    parser.add_argument('--n_negs', type=int, default=20, help="number of negative samples")
    parser.add_argument('--alpha', type=float, default=0.5, help="weight of word vector")
    parser.add_argument('--epoch', type=int, default=100, help="number of epochs")
    parser.add_argument('--mb', type=int, default=4096, help="mini-batch size")
    parser.add_argument('--ss_t', type=float, default=1e-5, help="subsample threshold")
    parser.add_argument('--conti', action='store_true', help="continue learning")
    parser.add_argument('--weights', action='store_true', help="use weights for negative sampling")
    parser.add_argument('--cuda', action='store_true', help="use CUDA")
    return parser.parse_args()


class PermutedSubsampledCorpus(Dataset):

    def __init__(self, datapath, ws=None):
        '''
            input:
                datapath: 
                ws: ??? word sample
            output:
        '''
        data = pickle.load(open(datapath, 'rb'))
        if ws is not None:
            self.data = []
            for iword, owords in data:
                if random.random() > ws[iword]:
                    self.data.append((iword, owords))
        else:
            self.data = data
        # print(self.data[:5])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        iword, owords = self.data[idx]
        return np.array(iword), np.array(owords)


def train(args):
    alpha = args.alpha
    idx2word = pickle.load(open(os.path.join(args.data_dir, 'idx2word.dat'), 'rb'))
    idx2sememe = pickle.load(open(os.path.join(args.data_dir, 'idx2sememe.dat'), 'rb'))
    widx2sidxs = pickle.load(open(os.path.join(args.data_dir, 'widx2sidxs.dat'), 'rb'))
    wordvec = np.load(os.path.join(args.data_dir, 'wordvec.npy'))
    wordvec = t.tensor(wordvec).float()
    # wc = pickle.load(open(os.path.join(args.data_dir, 'wc.dat'), 'rb')) #word count
    # wf = np.array([wc[word] for word in idx2word])  #word frequency
    # wf = wf / wf.sum()
    # ws = 1 - np.sqrt(args.ss_t / wf)
    # ws = np.clip(ws, 0, 1)
    vocab_size = len(idx2word)
    sememes_size = len(idx2sememe)
    print("sememe list has : {}".format(sememes_size))
    # weights = wf if args.weights else None
    weights = None
    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)
    model = Word2Vec(vocab_size=vocab_size, embedding_size=args.e_dim, wordvec= wordvec, sememes_size= sememes_size, widx2sidxs= widx2sidxs, alpha = alpha)
    modelpath = os.path.join(args.save_dir, '{}.pt'.format(args.name))
    sgns = SGNS(embedding=model, vocab_size=vocab_size, n_negs=args.n_negs, weights=weights)

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
    
    if os.path.isfile(modelpath) and args.conti:
        sgns.load_state_dict(t.load(modelpath))
    if args.cuda:
        sgns = sgns.cuda()
    optim = Adam(sgns.parameters(), lr=0.0001)
    optimpath = os.path.join(args.save_dir, '{}.optim.pt'.format(args.name))
    if os.path.isfile(optimpath) and args.conti:
        optim.load_state_dict(t.load(optimpath))
    for epoch in range(1, args.epoch + 1):
        dataset = PermutedSubsampledCorpus(os.path.join(args.data_dir, 'train.dat'))
        dataloader = DataLoader(dataset, batch_size=args.mb, shuffle=True)
        total_batches = int(np.ceil(len(dataset) / args.mb))
        pbar = tqdm(dataloader)
        pbar.set_description("[Epoch {}]".format(epoch))
        total_loss = 0
        count = 0
        for iword, owords in pbar:
            loss = sgns(iword, owords)
            optim.zero_grad()
            loss.backward()
            optim.step()
            pbar.set_postfix(loss=loss.item())
            total_loss += loss.item()
            count += 1
        print(total_loss / count)
    idx2vec = model.ivectors.weight.data.cpu().numpy()
    pickle.dump(idx2vec, open(os.path.join(args.data_dir, 'idx2vec.dat'), 'wb'))
    sidx2vec = model.svectors.weight.data.cpu().numpy()
    pickle.dump(sidx2vec, open(os.path.join("./vec/", args.name + '_sidx2vec.dat'), 'wb'))
    t.save(sgns.state_dict(), os.path.join(args.save_dir, '{}.pt'.format(args.name)))
    t.save(optim.state_dict(), os.path.join(args.save_dir, '{}.optim.pt'.format(args.name)))


if __name__ == '__main__':
    oldtime=datetime.datetime.now()
    train(parse_args())

    newtime=datetime.datetime.now()
    print(newtime)
    print('Runtime???%s'%(newtime-oldtime))
    print('Runtime???%sms'%(newtime-oldtime).microseconds)
    print('Runtime???%ss'%(newtime-oldtime).seconds)
