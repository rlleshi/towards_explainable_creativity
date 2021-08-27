import sys
import gensim
import pandas as pd

from argparse import ArgumentParser
from nltk import edit_distance
from nltk.corpus import wordnet as wn
from tqdm import tqdm


def parse_file(file):
    df = pd.read_csv(file, sep=';')
    return zip(df['w1'], df['w2'], df['w3'])


def has_solution(ground, result):
    has_solution = []
    for tup in zip(ground.values, result):
        # no solution
        if not tup[1]:
            has_solution.append(False)
            continue
        # no ground solution
        if str(tup[0]) == 'nan':
            continue

        if any(
            edit_distance(tup[0].strip(), sol) < 2 for sol in tup[1].split(', ')):
            has_solution.append(True)
        else:
            has_solution.append(False)
    return has_solution


def save_output(args, result):
    df = pd.read_csv(args.file, sep=';')
    df['Gensim'] = result
    df['has_solution'] = has_solution(df['wans'], result)
    df = df[['w1', 'w2', 'w3', 'wans', 'has_solution', 'Gensim']]

    df['Accuracy'] = ''
    df['Accuracy'].iloc[-1] = str(100*round(
        df['has_solution'].value_counts()[True] / (len(df['has_solution'])), 3)) + '%'
    df = df[['w1', 'w2', 'w3', 'wans', 'has_solution', 'Gensim', 'Accuracy']]

    df.to_excel(f'gensim_top{args.top_k}.xlsx')


def parse_args():
    parser = ArgumentParser(prog='get the closest words for triples using gensim')
    parser.add_argument('file', type=str, help='csv file frat')
    parser.add_argument(
        '--top-k',
        type=int,
        default=3,
        help='pick top-k embeddings')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    try:
        nlp_model = gensim.models.KeyedVectors.load('conceptNet', mmap='r')
    except FileNotFoundError:
        print('Make sure you have already converted the numberbatch model '
            'using the misc/number_batch_converter.py script')
        sys.exit()
    result = []
    nouns = list({x.name().split('.', 1)[0] for x in wn.all_synsets('n')})

    for triple in tqdm(parse_file(args.file)):
        try:
            triple_result = nlp_model.most_similar(
                positive=[node.strip() for node in triple], topn=args.top_k*10)

            filtered = [tr[0] for tr in triple_result if tr[0] in nouns][:args.top_k]
            result.append(', '.join(tr for tr in filtered))
        except KeyError as e:
            print(e)
            result.append([])

    save_output(args, result)


if __name__ == '__main__':
    main()
