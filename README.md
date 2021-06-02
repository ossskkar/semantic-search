# Semantic Search

This notebook presents the use of `SemanticSearch` class. `SemanticSearch` class makes use of pretrained BERT-based transformers `SentenceTransformers` specially tuned to perform semantic search. 

The process to perform a semantic search is as follow:

1. Encode corpus of text to a vector space.
2. Encode query text to the same vector space.
3. Find the sentences in the corpus that are most similar to the query text by means of cosine similarity score.


## 1. Installation & Setup

Install the required packages.

```
!pip install -U sentence-transformers
!pip install torch
```

## 2. Configuration

All required parameters to configure the model can be found and adjunsted in  `config.json`

```
import json

config = json.load(open('../semantic_search/config.json', 'r'))
for key in config.keys():
  print(f"{key}: '{config[key]}'")
```

## 3. SemanticSearch Class

Import and create an instance of SemanticSearch class

```
from model import SemanticSearch

ss = SemanticSearch()
```

## 4. Encode Corpus

All sentences in the corpus are encoded to a vector space and stored to a local pkl file for later use.

```
corpus_text = ['A man is eating food.',
          'A man is eating a piece of bread.',
          'The girl is carrying a baby.',
          'A man is riding a horse.',
          'A woman is playing violin.',
          'Two men pushed carts through the woods.',
          'A man is riding a white horse on an enclosed ground.',
          'A monkey is playing drums.',
          'A cheetah is running behind its prey.'
          ]

ss.update_corpus(corpus_text)

print(f"Corpus original text is stored at '{ss.config['CORPUS_TEXT_PATH']}'")
print(f"Corpus encoded text is stored at '{ss.config['CORPUS_ENCODED_PATH']}'")
```

## 5. Perform Queries

Given one or many sentences from the customer, find the most similar sentences in the corpus.

```
sentences = ['A man is eating pasta.', 
             'Someone in a gorilla costume is playing a set of drums.', 
             'A cheetah chases prey on across a field.']

for sentence in sentences:
  hits = ss.search(sentence)
  print('\nQuery: ', sentence)
  print('Hits:')
  for i, key in enumerate(hits.keys()):
    print(f'[#{i+1}] Sentence: {key} --> Similarity score: {hits[key]}')

```

## 6. Finetuning

The model can be finetuned by retraining with different datasets and training parameters. Training parameters can be found in `config["TRAIN"]`. To retrain the model, adjust the training parmeters accordingly and run the following script:

```
!python train.py
```

## Reference

More information on `SentenceTransformes` can be found [here](https://www.sbert.net/index.html)
