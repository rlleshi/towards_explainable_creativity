import re
import itertools
import logging
import os.path as osp

import numpy as np
import pandas as pd

from tqdm import tqdm
from argparse import ArgumentParser
from embeddings import GloveEmbedding
from sklearn.metrics.pairwise import cosine_similarity
from nltk import edit_distance


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
    parser.add_argument(
        '--merge-triple-tuple',
        action='store_true',
        help='consider the solutions of each tuple as part of the parent triple')
    args = parser.parse_args()
    return args


def parse_file(file):
    df = pd.read_excel(file, engine='openpyxl')
    df = df[df['FrAt'].notna()]
    nodes = [node if type(node) is str else None for node in df.FrAt]
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


def update_relations(relations, filtered_solutions):
    """Update relations based on the filtered solutions"""
    filtered_relations = []
    for tup in zip(relations.values, filtered_solutions):
        # no solution or no relation
        if (str(tup[0]) == 'nan') | (tup[1][0] == ' '):
            filtered_relations.append('')
            continue
        filtered_relation = ''
        splits = tup[0].split(' | ')
        for split in splits:
            if any(sol in split for sol in tup[1]):
                filtered_relation += f'{split} | '
        filtered_relations.append(filtered_relation)
    return filtered_relations


def update_has_solution(ground_solutions, filtered_solutions):
    has_solution = []
    for tup in zip(ground_solutions.values, filtered_solutions):
        # no solution
        if tup[1][0] == ' ':
            has_solution.append(False)
            continue
        # no ground solution
        if str(tup[0]) == 'nan':
            continue

        # ! why do we need levenshtein here? why won't simple equality work?
        if any(edit_distance(tup[0], sol) < 2 for sol in tup[1]):
            has_solution.append(True)
        else:
            has_solution.append(False)
    return has_solution


def separate_relations(relations):
    is_a, related_to, rest = [], [], []
    for relation in relations.values:
        is_a_temp = ''
        related_to_temp = ''
        rest_temp = ''

        for split in relation.split(' | '):
            if '"related_to"' in split:
                related_to_temp += f'{split} | '
            elif '"is_a"' in split:
                is_a_temp += f'{split} | '
            else:
                rest_temp += f'{split} | '
        is_a.append(is_a_temp)
        related_to.append(related_to_temp)
        rest.append(rest_temp)
    return is_a, related_to, rest


def update_df(args, embeddings, embeddings_best, filtered_solutions):
    df = pd.read_excel(args.file, engine='openpyxl')
    df = df[df['FrAt'].notna()]
    if args.merge_triple_tuple:
        # only triples come as end result
        df = df[df['FrAt'].apply(lambda x: len(x.split(', ')) == 3)]

    df['embeddings'] = embeddings
    df['top_embedding'] = embeddings_best
    df['solutions'] = [', '.join(sol for sol in solution) for solution in filtered_solutions]
    df['relation'] = update_relations(df['relation'], filtered_solutions)
    df['has_solution'] = update_has_solution(df['ground solution'], filtered_solutions)

    df['Accuracy'] = ''
    df['Accuracy'].iloc[-1] = str(100*round(
            df['has_solution'].value_counts()[True] / (len(df['has_solution'])), 3)) + '%'
    df['is_a'], df['related_to'], df['relation'] = separate_relations(df['relation'])

    df = df[['FrAt', 'ground solution', 'solutions', 'has_solution',
        'embeddings', 'top_embedding', 'relation', 'is_a', 'related_to', 'Accuracy']]
    out = (f'{osp.splitext(args.file)[0]}_top{args.top_k}'
        f'_merged-{args.merge_triple_tuple}_embeddings.xlsx')
    df.to_excel(out)
    print(f'Saved {out}')


def merge_tuples_to_triples(nodes, solutions):
    """"Incorportate the solutions of tuples into the corresponding triple parents"""
    new_nodes = []
    new_solutions = []
    for j in range(len(nodes)):
        splits = nodes[j].split(', ')

        if len(splits) == 3:
            new_nodes.append(nodes[j])
            new_solutions_temp = solutions[j] if solutions[j] is not None else ''

            for i in range(len(nodes)):
                for split in splits:
                    if split in nodes[i].split(', '):
                        tmp_solution = solutions[i] if solutions[i] is not None else ''
                        new_solutions_temp += ', ' + tmp_solution

            new_solutions.append(
                ', '.join(list(set(new_solutions_temp.split(', ')))))
    return new_nodes, new_solutions


def main():
    args = parse_args()
    nodes, solutions = parse_file(args.file)
    if args.merge_triple_tuple:
        nodes, solutions = merge_tuples_to_triples(nodes, solutions)

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
            msg = check_glove(g, pair)
            embd_msg += msg
            embd_value = re.findall('\d+\.\d+', msg)
            embd_best_temp[pair[1]] += float(embd_value[0]) if len(embd_value) > 0 else 0

        embd_best_temp = sorted(embd_best_temp.items(), key=lambda x: x[1], reverse=True)
        embeddings_best.append(embd_best_temp[0][0])
        embeddings.append(embd_msg)
        # top-k embeddings
        top_k = args.top_k if len(embd_best_temp) > args.top_k else len(embd_best_temp)
        filtered_solutions.append([embd_best_temp[i][0] for i in range(top_k)])

    update_df(args, embeddings, embeddings_best, filtered_solutions)

if __name__ == '__main__':
    logging.basicConfig(filename='embedding_logging.txt', level=logging.DEBUG)
    main()
