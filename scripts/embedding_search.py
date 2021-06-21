import re
import itertools
import logging

import numpy as np
import pandas as pd

from tqdm import tqdm
from argparse import ArgumentParser
from embeddings import GloveEmbedding
from sklearn.metrics.pairwise import cosine_similarity

# Motivation
# The Cosine Similarity between two word vectors provides an effective method
# for reassuring the linguistic or semantic similarity of the correspoinding words
# Sometimes, the nearest neighbors according to this metric reveal rare but relevant
# words that lie outside an average human's vocabulary.
# GloVe: https://nlp.stanford.edu/projects/glove/


def parse_args():
    parser = ArgumentParser(prog='check cosine similarity for word embeddings')
    parser.add_argument('file', type=str, help='csv file, rat or frat')
    parser.add_argument(
        '--top-k',
        type=int,
        default=10,
        help='pick top-k embeddings')
    args = parser.parse_args()
    return args


def parse_file(file):
    df = pd.read_excel(file, engine='openpyxl')
    try:
        nodes = [node if type(node) is str else None for node in df.FrAt]
    except AttributeError:
        nodes = [node if type(node) is str else None for node in df.RAT]
    solutions = [sol if type(sol) is str else None for sol in df.solutions]
    return nodes, solutions


def check_glove(g, pair):
    em1 = np.array(g.emb(pair[0])).reshape(1, -1)
    em2 = np.array(g.emb(pair[1])).reshape(1, -1)
    if np.any(em1) is None:
        return f'{pair[0]} - {pair[1]}: no embedding for {pair[0]}  |  '
    elif np.any(em2) is None:
        return f'{pair[0]} - {pair[1]}: no embedding for {pair[1]}  |  '
    return f'{pair[0]} - {pair[1]}: {round(cosine_similarity(em1, em2)[0][0], 3)}  |  '


def update_df(file, out, out_best, new_solutions):
    df = pd.read_excel(file, engine='openpyxl')
    df['embeddings'] = out
    df['top_embedding'] = out_best
    df['solutions'] = [', '.join(sol for sol in solution) for solution in new_solutions ]
    try:
        df = df[['FrAt', 'ground solution', 'solutions', 'has_solution', 'embeddings', 'top_embedding', 'relation', 'Accuracy']]
    except KeyError:
        df = df[['RAT', 'ground_solution', 'solutions', 'has_solution', 'embeddings', 'top_embedding', 'relation', 'Accuracy']]
    df.to_excel(file.strip('.xlsx') + '_embeddings.xlsx')
    df.to_json(file.strip('.xlsx') + '_embeddings.json', indent=4)
    print('Saved output')


def main():
    args = parse_args()
    nodes, solutions = parse_file(args.file)
    g = GloveEmbedding('common_crawl_840', d_emb=300, show_progress=True)
    embeddings = []
    embeddings_best = []
    filtered_solutions = []

    for i in tqdm(range(len(solutions))):
        if solutions[i] is None:
            embeddings.append('')
            embeddings_best.append('')
            filtered_solutions.append([' '])
            continue
        logging.info('# Examining Nodes {} | Solutions: {}...'.format(nodes[i], solutions[i]))

        embd_msg = ''
        sols = solutions[i].split(', ')
        embd_best_temp = {sol: 0 for sol in sols}

        # cartesian product between 3 nodes of queries & solutions
        for pair in itertools.product(nodes[i].split(', '), sols):
            logging.info('## Examining {}'.format(pair))
            res = check_glove(g, pair)
            embd_msg += res
            embd_value = re.findall('\d+\.\d+', res)
            embd_best_temp[pair[1]] += float(embd_value[0]) if len(embd_value) > 0 else 0

        # ! Fix order (desc not asc)
        embd_best_temp = sorted(embd_best_temp.items(), key=lambda x: x[1])
        embeddings_best.append(embd_best_temp[0][0])
        embeddings.append(embd_msg)
        # top-k embeddings
        top_k = args.top_k if len(embd_best_temp) > args.top_k else len(embd_best_temp)
        filtered_solutions.append([embd_best_temp[i][0] for i in range(top_k)])

    update_df(args.file, embeddings, embeddings_best, filtered_solutions)

if __name__ == '__main__':
    logging.basicConfig(filename='embedding_logging.txt', level=logging.DEBUG)
    main()
