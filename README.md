# r_m_thesis

### Thesis: Towards Explainable Creativity: Tackling The Remote Association Test With Knowledge Graphs

15.02.2021 - 15.07.2021

#### ConceptNet search local
Given a triple of concepts e.g., (question, reply, solution) and a ground solution e.g., statement, calculate:

- *solutions*: list of solutions based on conceptnet using [this](https://github.com/ldtoolkit/conceptnet-lite) python library. A solution is a node from three queries that result in the same node for every member of a triple.
- *has_solution*: boolean, whether the triple has a solution or not
- *relation*: bidirectional relation from each member of a triple to it's solution
- *relation_to_solution*: unidirectional relation from each member of a triple to it's solution
- *relation_from_solution*: unidirectional relation from the solution to each member of a triple
- *accuracy*: accuracy calculated against *ground_solution*

For frat: look at non-compound concepts
For rat: look at compound concepts

#### Embedding search
The Cosine Similarity between two word vectors provides an effective method for reassuring the linguistic or semantic similarity of the correspoinding words. Sometimes, the nearest neighbors according to this metric reveal rare but relevant words that lie outside an average human's vocabulary.

Here all the nodes are checked against the solutions and a cosine distance is noted.

