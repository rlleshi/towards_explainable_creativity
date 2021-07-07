import numpy as np
import pandas as pd

from tqdm import tqdm
from argparse import ArgumentParser
from embeddings import GloveEmbedding
from sklearn.metrics.pairwise import cosine_similarity
from nltk import edit_distance
from nltk.corpus import wordnet as wn


# def tests():
#     from embeddings.embedding import Embedding
#     path_glove = '/home/rlleshi/.embeddings/glove/common_crawl_840:300.db'
#     db = Embedding.initialize_db(path_glove)
#     w = 'care'
#     cursor = db.cursor()
#     query = cursor.execute('select word from embeddings LIMIT 100').fetchall()
#     print(query)
#     q = cursor.execute('select emb from embeddings where word = :word', {'word': w}).fetchone()
#     print(q.fetchone()[0])
#     q = cursor.execute('select count(*) from embeddings')
#     print(q.fetchone()[0])
#     db.close()

CACHE = {}

def parse_file(file):
    df = pd.read_csv(file, sep=';')
    return zip(df['w1'], df['w2'], df['w3'])


def parse_args():
    parser = ArgumentParser(prog='get the closest words for triples')
    parser.add_argument('file', type=str, help='csv file, rat or frat')
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='threshold for cosine similarity')
    parser.add_argument(
        '--top-k',
        type=int,
        default=10,
        help='pick top-k embeddings')
    args = parser.parse_args()
    return args


def check_glove(g, node, noun):
    em1 = np.array(g.emb(node)).reshape(1, -1)
    if np.any(em1) is None:
        return 0

    if CACHE.get(noun, None) is not None:
        em2 = CACHE[noun]
    else:
        em2 = np.array(g.emb(noun)).reshape(1, -1)
        if np.any(em2) is None:
            return 0
        CACHE[noun] = em2

    return round(cosine_similarity(em1, em2)[0][0], 3)


def save_output(args, result):
    df = pd.read_csv(args.file, sep=';')
    df['Top Embeddings'] = result
    df = df[['w1', 'w2', 'w3', 'wans', 'Top Embeddings']]
    df.to_excel(f'top_embeddings_{args.threshold}thr_{args.top_k}topk.xlsx')


def main():
    args = parse_args()
    g = GloveEmbedding('common_crawl_840', d_emb=300, show_progress=True)
    nouns = list({x.name().split('.', 1)[0] for x in wn.all_synsets('n')})
    global CACHE
    result = []

    for triple in tqdm(parse_file(args.file)):
        triple_result = []

        for node in triple:
            temp = []
            for noun in nouns:
                if check_glove(g, node.strip(), noun) > args.threshold:
                    temp.append(noun)
            triple_result.append(temp)

        intersection = list(set(triple_result[0]) & set(triple_result[1]) & set(triple_result[2]))
        result.append(', '.join(intersection[:args.top_k]))

    save_output(args, result)


if __name__ == '__main__':
    main()