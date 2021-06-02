
from sentence_transformers import SentenceTransformer, util

import json
import os
import torch

class Encoder:

  def __init__(self, config):
    self.config = config
    
    if self.model_download:
      self.download()
      self.save()
    else:
      self.load()

  @property 
  def model_download(self):
    return not os.path.isfile(self.config['MODEL_PATH']) or self.config['MODEL_DOWNLOAD'] 

  def load(self):
    self.model = SentenceTransformer(self.config['MODEL_PATH'])

  def download(self):
    self.model = SentenceTransformer(self.config['MODEL_NAME'])
    
  def save(self):
    self.model.save(self.config['MODEL_PATH'])

  def encode(self, corpus):
    return self.model.encode(corpus, convert_to_tensor=True)


class Corpus:

  def __init__(self, config):
    self.config = config
    self.load()

  @property
  def corpus_text_missing(self):
    return not os.path.isfile(self.config['CORPUS_TEXT_PATH'])

  @property
  def corpus_encoded_missing(self):
    return not os.path.isfile(self.config['CORPUS_ENCODED_PATH'])

  def load(self):
    self.corpus_text = None if self.corpus_text_missing else torch.load(self.config['CORPUS_TEXT_PATH'])
    self.corpus_encoded = None if self.corpus_encoded_missing else torch.load(self.config['CORPUS_ENCODED_PATH'])

  def save(self):
    torch.save(self.corpus_text, self.config['CORPUS_TEXT_PATH'])
    torch.save(self.corpus_encoded, self.config['CORPUS_ENCODED_PATH'])

  def update(self, corpus_text, corpus_encoded):
    self.corpus_text = corpus_text
    self.corpus_encoded = corpus_encoded
    self.save()


class SemanticSearch:

  def __init__(self, config_path=None):
    config_path = config_path if config_path else 'config.json'
    with open(config_path, 'r') as file:
      self.config = json.load(file)

    self.encoder = Encoder(self.config)
    self.corpus = Corpus(self.config)

  def update_corpus(self, corpus_text):
    corpus_encoded = self.encoder.encode(corpus_text)
    self.corpus.update(corpus_text, corpus_encoded)

  def search(self, sentence, top_k=3):
    sentence_encoded = self.encoder.encode(sentence)

    cos_scores = util.pytorch_cos_sim(sentence_encoded, self.corpus.corpus_encoded)[0]
    hits = torch.topk(cos_scores, k=top_k)
    hits = {self.corpus.corpus_text[idx]: round(float(score),2) for score, idx in zip(hits[0], hits[1])}

    return hits
