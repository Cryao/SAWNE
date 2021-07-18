import pickle
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='sgns', help="model name")
    parser.add_argument('--data_dir', type=str, default='./data/', help="data directory path")
    parser.add_argument('--result', type=str, default='result.txt', help="predict result file")
    return parser.parse_args()


def Scorer_topk(args, topk):
    word2idx = pickle.load(open(os.path.join(args.data_dir, 'word2idx.dat'), 'rb'))
    # idx2word = pickle.load(open(os.path.join(args.data_dir, 'idx2word.dat'), 'rb'))
    word2synset = pickle.load(open(os.path.join(args.data_dir, 'word2synset.dic'), 'rb'))
    result_file = open(os.path.join("./result/", args.result), 'r')
    results = result_file.readlines()
    count = 0
    for i in range(0, len(results), 2):
        word = results[i].strip()
        synsets = results[i + 1].strip().split()
        for synset in synsets[:topk]:
            if synset in word2synset[word]:
                count += 1
                break
    print('Top{} : {}'.format(topk, count / len(results) * 2))


if __name__ == '__main__':
    Scorer_topk(parse_args(), 5)
    # Scorer_topk(parse_args(), 10)
    # Scorer_topk(parse_args(), 50)
    # Scorer_topk(parse_args(), 100)
    # Scorer_topk(parse_args(), 20)
    # Scorer_topk(parse_args(), 30)
    # Scorer_topk(parse_args(), 40)
    # Scorer_topk(parse_args(), 1)
    # Scorer_topk(parse_args(), 3)