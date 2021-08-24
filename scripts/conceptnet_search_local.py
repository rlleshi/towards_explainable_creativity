import time
import os
import logging
import pandas as pd
import conceptnet_lite
import multiprocessing

from itertools import repeat
from argparse import ArgumentParser
from multiprocessing import Pool, Manager, freeze_support
from conceptnet_lite import Label, edges_between, edges_for

#
# ! Multiprocessing doesn't work because the conceptnet_lite library did not implement it
START_TIME = time.time()
#

def get_conceptnet():
    conceptnet_lite.connect('conceptnet_database.db')

def save_csv(args, OUTPUT):
    output = pd.DataFrame(list(OUTPUT))
    f_name = args.rat_frat + '_' + str(args.check) + '_conceptnet_search.xlsx'
    dIr = os.path.join('output', args.rat_frat)
    if not os.path.exists(dIr):
        os.mkdir(dIr)
    output.to_excel(os.path.join(dIr, f_name), index=False)

def parse_args():
    parser = ArgumentParser(prog='concepnet search for rat or frat')
    parser.add_argument('src', type=str, help='rat or frat .csv file')
    parser.add_argument(
        '--rat-frat',
        type=str,
        default='frat',
        choices=['rat', 'frat'],
        help='rat or frat file')
    parser.add_argument(
        '--check',
        type=int,
        default=3,
        choices=[2,3],
        help='check for triples or doubles'
    )
    args = parser.parse_args()
    return args

def get_nodes_frat(node):
    """ Given a node, get all the other nodes related to it
        The search is performed by looking at all the edges related to a particular node.

        Returns:
            dictionary of form:
                {"pick_someone's_brain": ['related_to'], 'blindly': ['related_to'], ...  'cross_purpose': ['related_to']}
    """
    try:
        # start = lambda e: (e.start.text, e.relation.name, 'from')
        # end = lambda e: (e.end.text, e.relation.name, 'to')
        # edges starting from (our node)
        current = [(e.start.text, e.relation.name)
                    for e in edges_for(Label.get(text=node, language='en').concepts, same_language=True)
                       if e.start.text not in [node]]
        [current.append((e.end.text, e.relation.name))
            for e in edges_for(Label.get(text=node, language='en').concepts, same_language=True)
                   if e.end.text not in [node]]

        result = {}
        for tup in list(set(current)):
            if tup[0] not in result:
                result[tup[0]] = list()
                result[tup[0]].append(tup[1])
            else:
                result[tup[0]].append(tup[1])
        return result
    except Exception as e:
        msg = '!!!   No label for the node "{}"... Are you sure the spelling is correct?'.format(node)
        logging.exception(msg)
        print(msg, e)
        return {}

def get_nodes_rat(word):
    """ Given a word, get all the compound words related to it as well as their relation name
        Compound words are basically being identified by the underscore (_)
    """
    result = []
    relation = []
    for e in edges_for(Label.get(text=word).concepts, same_language=True):
        if (e.start.text.find('_') != -1) & (e.start.text.find(word) != -1):
            result.append(e.start.text.replace(word, '').strip('_'))
            relation.append(e.relation.name)
        if (e.end.text.find('_') != -1) & (e.end.text.find(word) != -1):
            result.append(e.end.text.replace(word, '').strip('_'))
            relation.append(e.relation.name)

    joint_result = []
    for i in range(len(result)):
        if result[i].find('_') != -1:
            words = result[i].split('_')
            for word in words:
                if word != '':
                    joint_result.append((word, relation[i]))
        else:
            joint_result.append((result[i], relation[i]))

    final_result = {}
    for tup in list(set(joint_result)):
        if tup[0] not in final_result:
            final_result[tup[0]] = list()
            final_result[tup[0]].append(tup[1])
        else:
            final_result[tup[0]].append(tup[1])
    return final_result

def check_for(relation_dict, check_for):
    results = [set(relation_dict[key].keys()) for key in relation_dict.keys()]
    if 3 == check_for:
        yield results[0] & results[1] & results[2]
    if 2 == check_for:
        yield results[0] & results[1]
        yield results[0] & results[2]
        yield results[1] & results[2]

def get_output(result, query, relation_dict, ground_solution, has_solution):
    solutions = [sol for sol in result] if result else []
    relations = [] # both directions
    to_solution = [] # node -> solution
    from_solution = [] # solution -> node

    # build a relationship message for: (1) node, (2) relation (3) solution
    # For example:
    #
    # cues: antlers, doe, fawn
    # relation: related_to
    # solution: deer
    # relationship message: antler is related_to deer, doe is related_to to deer, fawn is related_to to deer
    for node in query:
        for sol in solutions:
            rel = ', '.join(relation_dict[node][sol.strip()]) # get the relationships for each cue and solution
            relations.append(node + ' is "'+ rel + '" to ' + sol)

    for node in query:
        for sol in solutions:
            for e in edges_between(Label.get(text=cue, language='en').concepts, Label.get(text=sol, language='en').concepts):
                to_solution.append(e.start.text + ' "' + e.relation.name + '" ' + e.end.text)

    for solution in solutions:
        for node in query:
            for e in edges_between(Label.get(text=solution, language='en').concepts, Label.get(text=node, language='en').concepts):
                from_solution.append(e.start.text + ' "' + e.relation.name + '" ' + e.end.text)

    return {'FrAt': ', '.join(query),
            'ground solution': ground_solution,
            'solutions': ', '.join(solutions),
            'has_solution': has_solution,
            'relation': ' | '.join(relations),
            'relation_to_solution': ' | '.join(to_solution),
            'relation_from_solution': ' | '.join(from_solution)}

def compute(items):
    index, query, df, args, OUTPUT, ACCURACY = items
    get_nodes = get_nodes_rat if args.rat_frat == 'rat' else get_nodes_frat
    solution = df.iloc[index].wans
    relation_dict = {}
    msg = 'Examining {}. Timestamp: {} min'.format(query, round((time.time()-START_TIME)/60, 2))
    logging.info(msg)
    print(msg)

    for node in query:
        relation_dict[node] = get_nodes(node)

    # the format of the results dictionary at this point would be
    #
    # {
    #  'question': {"pick_someone's_brain": ['related_to'], 'blindly': ['related_to'], ... 'cross_purpose': ['related_to']},
    #  'reply': {'repone': ['related_to'], ... 'sentences': ['related_to']},
    #  'solution': {'solutionism': ['derived_from', 'related_to'],... 'exhibit': ['related_to']}
    # }

    for result in check_for(relation_dict, args.check):
        ACCURACY['total'] += 1
        has_solution = any(solution.lower().strip() in node for node in result)
        if has_solution:
            ACCURACY['tp'] += 1
        OUTPUT.append(get_output(result, query, relation_dict, solution, has_solution))

def main():
    args = parse_args()
    get_conceptnet()
    logging.basicConfig(filename=args.rat_frat + '_conceptnet_search_logs.txt', level=logging.DEBUG)

    if args.rat_frat == 'rat': # csvs differ
        df = pd.read_csv(args.src)
    else:
        df = pd.read_csv(args.src, sep=';')
    # queries format
    # [['question', 'reply', 'solution'], ... ['fault', 'incorrect', 'unjust']]
    queries = df.w1 + ' ' + df.w2 + ' ' + df.w3
    queries = [list(map(lambda x: x.lower(), filter(len, line.split(' ')))) for line in queries]
    pool = Pool(multiprocessing.cpu_count() - 1 or 1)
    pool.map(
            compute,
            zip(range(0, len(queries)),
                queries,
                repeat(df),
                repeat(args),
                repeat(OUTPUT),
                repeat(ACCURACY)))

    OUTPUT.append({'Accuracy': str(round(100*ACCURACY['tp']/ACCURACY['total'], 2)) + '%'})
    save_csv(args, OUTPUT)

if __name__ == '__main__':
    manager = Manager()
    OUTPUT = manager.list()
    ACCURACY= manager.dict()
    ACCURACY['total'] = 0
    ACCURACY['tp'] = 0
    main()
