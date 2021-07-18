import os
import codecs
import pickle
import argparse
import json 
import numpy as np 
from sklearn.model_selection import train_test_split
from collections import defaultdict

# os.chdir(r'F:\VScode\Work2\Baseline\data')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type= str, default='./data/', help= "data directory path")
    parser.add_argument('--word2synset', type= str, default='./data/word2synset.dic', help= "word2synset dict file path")
    parser.add_argument('--hyperRelations', type= str, default='./data/hyperRelations.dic', help= "hyperRelation dict file path")
    parser.add_argument('--simRelations', type= str, default='./data/simRelations.dic', help= "simRelation dict file path")
    parser.add_argument('--hownet', type= str, default= './data/HowNet_Data', help= "HowNet file path")
    parser.add_argument('--wordvec', type= str, default= './data/embedding.txt', help= "word vec file path")
    parser
    return parser.parse_args()

class Preprocess(object):

    def __init__(self, word2synset_file = None, hyperRelations_file = None, simRelations_file = None, vocab = None, numOfpos = 5, unk = '<UNK>', data_dir = './data/'):
        self.word2synset = pickle.load(open(word2synset_file, 'rb'))
        self.synset2word = defaultdict(set)
        for word in self.word2synset:
            synsets = self.word2synset[word]
            for synset in synsets:
                self.synset2word[synset].add(word)
        self.hyperRelations = pickle.load(open(hyperRelations_file, 'rb'))
        self.hyponRelations = defaultdict(set)
        for synset in self.hyperRelations:
            hypers = self.hyperRelations[synset]
            for syn in hypers:
                self.hyponRelations[syn].add(synset)
        self.simRelations = pickle.load(open(simRelations_file, 'rb'))
        self.vocab = vocab
        if not vocab:
            self.vocab = list(self.word2synset.keys())
        self.numOfpos = numOfpos
        self.unk = unk
        self.data_dir = data_dir

    def init_sememes(self, filepath):
        self.sememes = set()
        self.max_sememe_len = 0
        self.word2sememes = defaultdict(set)
        hownet_file = open(filepath, 'r', encoding= 'utf-8')
        hownet = hownet_file.readlines()
        for i in range(0, len(hownet), 2):
            word = hownet[i].strip()
            sememes = set(hownet[i+1].strip().split())
            self.word2sememes[word] |= sememes
            self.max_sememe_len = max(self.max_sememe_len, len(self.word2sememes[word]))
            self.sememes |= sememes
        print("word max has {} sememes".format(self.max_sememe_len))
        self.idx2sememe = list(self.sememes)
        self.sememe2idx = {self.idx2sememe[idx] : idx for idx, _ in enumerate(self.idx2sememe)}
        pickle.dump(self.idx2sememe, open(os.path.join(self.data_dir, 'idx2sememe.dat'), 'wb'))
        pickle.dump(self.sememe2idx, open(os.path.join(self.data_dir, 'sememe2idx.dat'), 'wb'))


    def find_pos(self, iword):
        '''
        find in hyperRelations
        '''
        tmp_pos = []
        isynset = self.word2synset[iword]
        syn_queue = []
        for synset in isynset:
            if synset in self.hyperRelations:
                syn_queue += list(self.hyperRelations[synset])
        while syn_queue:
            size = len(syn_queue)
            for i in range(size):
                cur_synset = syn_queue.pop(0)
                if cur_synset in self.hyperRelations:
                    syn_queue.extend(list(self.hyperRelations[cur_synset]))
                if cur_synset in self.synset2word:
                    tmp_pos.extend(list(self.synset2word[cur_synset]))
            if len(tmp_pos) >= self.numOfpos:
                break
        owords = tmp_pos[:self.numOfpos]
        return iword, owords + [self.unk] * max(self.numOfpos - len(owords), 0)
    
    def find_pos_(self, iword):
        '''
        find in simRelations and hyperRelations
        '''
        tmp_pos = []
        syn_queue = []
        isynsets = self.word2synset[iword]
        for synset in isynsets:
            if synset in self.hyperRelations:
                syn_queue += list(self.hyperRelations[synset])
            if synset in self.simRelations:
                syn_queue += list(self.simRelations[synset])
        while syn_queue:
            size = len(syn_queue)
            for i in range(size):
                cur_synset = syn_queue.pop(0)
                if cur_synset in self.hyperRelations:
                    syn_queue.extend(list(self.hyperRelations[cur_synset]))
                if cur_synset in self.synset2word:
                    tmp_pos.extend(list(self.synset2word[cur_synset]))
            if len(tmp_pos) >= self.numOfpos:
                break
        owords = tmp_pos[:self.numOfpos]
        return iword, owords + [self.unk] * max(self.numOfpos - len(owords), 0)
    
    def build(self, ):
        # word include sememes
        self.idx2word = [self.unk] + self.vocab # + self.sememes
        self.word2idx = {self.idx2word[idx] : idx for idx, _ in enumerate(self.idx2word)}
        self.widx2sidxs = list()
        for idx, word in enumerate(self.idx2word):
            sememes = self.word2sememes[word]
            sidxs = [self.sememe2idx[sememe] for sememe in sememes]
            self.widx2sidxs.append(sidxs + [0] * (self.max_sememe_len - len(sidxs)))
        pickle.dump(self.idx2word, open(os.path.join(self.data_dir, 'idx2word.dat'), 'wb'))
        pickle.dump(self.word2idx, open(os.path.join(self.data_dir, 'word2idx.dat'), 'wb'))
        pickle.dump(self.widx2sidxs, open(os.path.join(self.data_dir, 'widx2sidxs.dat'), 'wb'))
    
    def prepare_data(self, ):
        '''
        add sememes
        '''
        step = 0
        data = []
        not_pos = 0
        for word in self.vocab:
            step += 1
            if not step % 1000:
                print("working on {}kth word".format(step // 1000), end = '\r')
            iword, owords = self.find_pos_(word)
            # print("iword: {}   owords:{}".format(iword, ' '.join(owords)))
            if all([self.unk == item for item in owords]):
                not_pos += 1
            # isememes = [self.sememe2idx[sememe] for sememe in self.word2sememes[iword]]
            # isample = [self.word2idx[iword]] + isememes + [0] * (self.max_sememe_len - len(isememes))
            # osamples = []
            # for word in owords:
            #     osememes = [self.sememe2idx[sememe] for sememe in self.word2sememes[word]]
            #     osample = [self.word2idx[word]] + osememes + [0] * (self.max_sememe_len - len(osememes))
            #     osamples.append(osample)
            # data.append((isample, osamples))
            data.append((self.word2idx[iword], [self.word2idx[oword] for oword in owords]))
            # print(data[-1])
        print("")
        print("threre are {} examples in data, {} has not pos example.".format(len(data), not_pos))
        train_data, test_data = train_test_split(data, test_size = 0.1, random_state = 42)
        pickle.dump(train_data, open(os.path.join(self.data_dir, 'train.dat'), 'wb'))
        pickle.dump(test_data, open(os.path.join(self.data_dir, 'test.dat'), 'wb'))
        print("prepare data done")

    # def prepare_data()
    
    def prepare_vec(self, filepath):
        self.word2vec = dict()
        wordvec_file = open(filepath, 'r', encoding='utf-8')
        _, dim = map(int, wordvec_file.readline().strip().split())
        self.parameter = np.zeros(dim)
        print('word vec dim: {}'.format(dim))
        for line in wordvec_file:
            line = line.strip().split()
            word, vec = line[0], line[1:]
            if word in self.vocab:
                self.word2vec[word] = list(map(float, vec))
        no_apeared_words = set()
        for _, word in enumerate(self.idx2word[1:]):
            if word in self.word2vec:
                vec = np.array(self.word2vec[word])
                self.parameter = np.vstack((self.parameter, vec))
            else:
                print('{} has not apeared'.format(word))
                no_apeared_words.add(word)
                vec = np.random.rand(dim)
                self.parameter = np.vstack((self.parameter, vec))
        np.save(os.path.join(self.data_dir, 'wordvec.npy'), self.parameter)
        print("word parameter size: {}".format(self.parameter.shape))
        print("{} has not apeared".format(len(no_apeared_words)))

if __name__ == '__main__':
    args = parse_args()
    preprocess = Preprocess(word2synset_file= args.word2synset, hyperRelations_file= args.hyperRelations, simRelations_file= args.simRelations)
    preprocess.init_sememes(args.hownet)
    preprocess.build()
    preprocess.prepare_data()
    preprocess.prepare_vec(args.wordvec)