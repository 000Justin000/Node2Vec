## Julia Node2Vec Implementation
#### Author: Junteng Jia

This is a Julia implementation of [Node2Vec](https://cs.stanford.edu/people/jure/pubs/node2vec-kdd16.pdf) which is built upon [LightGraphs](https://github.com/JuliaGraphs/LightGraphs.jl).
On a high level, Node2Vec first generate sequence of vertices with biased random walks controlled by two parameters p and q, then it uses the Word2Vec machinery to compute vertex embedding.

### Installation

```julia
] add https://github.com/000Justin000/Node2Vec
```

#### Basic Usage

First, generate vertex sequences with biased random walks.
```julia
walks = simulate_walks(g, num_rounds, len, p, q)
"""
Args:
  g: a LightGraph object
  num_rounds: number of round of simulation
  len: maximum random walk length
  p: return hyperparameter
  q: inout hyperparameter
"""
```

Once you have the sequences of vertices, generate vertex embeddings by:
```julia
model = learn_embeddings(walks, ndim)
"""
Args:
  walks: sequence of vertices
  ndim: dimension for vertex embeddings
"""
```

If you have any questions, please email to [jj585@cornell.edu](mailto:jj585@cornell.edu).
