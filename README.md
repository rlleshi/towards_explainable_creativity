# Towards Explainable Creativity: Tackling The Remote Association Test With Knowledge Graphs

15.02.2021 - 15.09.2021

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
