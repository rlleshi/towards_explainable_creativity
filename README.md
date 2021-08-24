# Towards Explainable Creativity: Tackling The Remote Association Test With Knowledge Graphs

*TimeFrame: 15.02.2021 - 15.09.2021*

# Results

| Model | Accuracy_1 | Accuracy_2 | Accuracy_3 |
| ------ | ------ | ------ | ------ |
| conceptnet-search-local | frat_3_depth_1: *27.08%*; frat_3_depth_2: *83.33%*| frat_2,3_depth_1: *62.5%*; frat_2,3_depth_2: *86.98%* | frat_2_depth_1: *62.5%*; frat_2_depth_2: *88.19%* |
| embedding-search-local | frat_3_depth_1: *25.0%*; frat_top_3_depth_2: *54.2%* | frat_2,3_depth_1: *62.5%*; frat_top_5_depth_2: *64.6%* | frat_2_depth_1: *37.5%*; frat_top_10_depth_2: *70.8%* |
| embedding-search-general | for running the methods as a seldon-core service used in the future |
| genism-search | frat_top_3: *16.7%* | frat_top_5: *18.8%* | frat_top_10: *31.2%* |

The code for this thesis focuses on the following

## ConceptNet search local
Given a triple of concepts e.g., (question, reply, solution) and a ground solution e.g., statement, calculate:

- *solutions*: list of solutions based on conceptnet using [this](https://github.com/ldtoolkit/conceptnet-lite) python library. A solution is a node from three queries that result in the same node for every member of a triple.
- *has_solution*: boolean, whether the triple has a solution or not
- *relation*: bidirectional relation from each member of a triple to it's solution
- *relation_to_solution*: unidirectional relation from each member of a triple to it's solution
- *relation_from_solution*: unidirectional relation from the solution to each member of a triple
- *accuracy*: accuracy calculated against *ground_solution*

For frat: look at non-compound concepts (focus of the thesis)

For rat: look at compound concepts

Moreover, model the explanations (*relation*) according to *templates.txt*

## Embedding search
The Cosine Similarity between two word vectors provides an effective method for reassuring the linguistic or semantic similarity of the correspoinding words. Sometimes, the nearest neighbors according to this metric reveal rare but relevant words that lie outside an average human's vocabulary.

Here all the nodes are checked against the solutions and a cosine distance is noted. Afterwards, solutions are filtered according to a threshold.

### Embedding general search
For every noun in the English language (65k taken from WordNet), get the embedding and thereafter find the cosine similarity with our triples. Finally, perform intersection between triple's solutions.

## Gensim search
Find the intersection of the queries based on [CenceptNet Numberbatch](https://github.com/commonsense/conceptnet-numberbatch). Find top3, top5, top10 solutions.

In order to use the script you must first download the conceptnet-numberbatch model (link in `resources/`), unzip the model and convert it using the `misc/number_batch_converter.py` script.
