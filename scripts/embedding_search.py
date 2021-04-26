import os
import re
import time
import itertools

import numpy as np
import pandas as pd

from argparse import ArgumentParser
from embeddings import GloveEmbedding, FastTextEmbedding, KazumaCharEmbedding, ConcatEmbedding
from sklearn.metrics.pairwise import cosine_similarity

######
# The Cosine Similarity between two word vectors provides an effective method for reassuring the linguistic or semantic similarity
# of the correspoinding words. Sometimes, the nearest neighbors according to this metric reveal rare but relevant words that lie outside
# an average human's vocabulary.
# GloVe: https://nlp.stanford.edu/projects/glove/
#
######

def parse_args():
    parser = ArgumentParser(prog='check cosine similarity for word embeddings')
    parser.add_argument('File', metavar='file',
                        type=str, help='csv file, rat or frat')
    args = parser.parse_args()
    return args

def parse_file(file):
    df = pd.read_excel(file)
    try:
        nodes = [node if type(node) is str else None for node in df.FrAt]
    except AttributeError:
        nodes = [node if type(node) is str else None for node in df.RAT]
    solutions = [sol if type(sol) is str else None for sol in df.solutions]
    return nodes, solutions


def check_glove(g, pair):
    em1 = np.array(g.emb(pair[0])).reshape(1, -1)
    em2 = np.array(g.emb(pair[1])).reshape(1, -1)
    if np.any(em1) is None: return '{} - {}: no embedding for {}  |  '.format(pair[0], pair[1], pair[0])
    elif np.any(em2) is None: return '{} - {}: no embedding for {}  |  '.format(pair[0], pair[1], pair[1])
    return '{} - {}: {}  |  '.format(pair[0], pair[1], round(cosine_similarity(em1, em2)[0][0], 3))

def update_df(file, out, out_best):
    df = pd.read_excel(file)
    df['embeddings'] = out
    df['top_embedding'] = out_best
    try:
        df = df[['FrAt', 'ground solution', 'solutions', 'has_solution', 'embeddings', 'top_embedding', 'relation', 'Accuracy']]
    except KeyError:
        df = df[['RAT', 'ground_solution', 'solutions', 'has_solution', 'embeddings', 'top_embedding', 'relation', 'Accuracy']]
    df.to_excel(file.strip('.xlsx') + '_embeddings.xlsx')
    df.to_json(file.strip('.xlsx') + '_embeddings.json', indent=4)
    print('Saved output')

if __name__ == '__main__':
    args = parse_args()
    nodes, solutions = parse_file(args.File)
    start_time = time.time()
    g = GloveEmbedding('common_crawl_840', d_emb=300, show_progress=True)
    out = []
    out_best = []

    for i in range(len(solutions)):
        if solutions[i] is None:
            out.append('')
            out_best.append('')
            continue
        print('# Examining Nodes {} | Solutions: {}...'.format(nodes[i], solutions[i]))

        result = ''
        sols = solutions[i].split(', ')
        best_embd = {sol: 0 for sol in sols}

        for pair in itertools.product(nodes[i].split(', '), sols): # cartesian product
            print('## Examining {}'.format(pair))
            res = check_glove(g, pair)
            result += res
            val = re.findall('\d+\.\d+', res)
            best_embd[pair[1]] += float(val[0]) if len(val) > 0 else 0

        mAx = max(best_embd.values())
        for k, v in best_embd.items():
            if v == mAx: out_best.append(k)
        out.append(result)

        print('Timestamp: {}'.format(round(time.time()-start_time, 2)))

    update_df(args.File, out, out_best)
