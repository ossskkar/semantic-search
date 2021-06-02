"""
DEVELOPMENT EVALUATORS
Three evaluators are implemented, that evaluate the model on:

    1. Duplicate Questions pair classification,
    2. Duplicate Questions Mining
    3. Duplicate Questions Information Retrieval


CLASSIFICATION EVALUATOR
Given (quesiton1, question2), is this a duplicate or not?
The evaluator will compute the embeddings for both questions and then compute
a cosine similarity. If the similarity is above a threshold, we have a duplicate.

MINING EVALUATOR
Given a large corpus of questions, identify all duplicates in that corpus.

INFORMATION RETRIEVAL
Given a question and a large corpus of thousands questions, find the most relevant (i.e. duplicate) question
in that corpus.

"""

from sentence_transformers.evaluation import BinaryClassificationEvaluator
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers.evaluation import ParaphraseMiningEvaluator
from sentence_transformers.evaluation import SequentialEvaluator

import csv
import os 
import random

class Evaluator:

    def __init__(self, config):
        self.config = config


    def classification_evaluator(self):
        dev_sentences1 = []
        dev_sentences2 = []
        dev_labels = []

        dataset_path = os.path.join(self.config['TRAIN']['DATASET_PATH'], self.config['TRAIN']['DATASET_CLASSIFICATION_PATH'])
        with open(dataset_path, encoding='utf8') as fIn:
            reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
            for row in reader:
                dev_sentences1.append(row['question1'])
                dev_sentences2.append(row['question2'])
                dev_labels.append(int(row['is_duplicate']))

        return BinaryClassificationEvaluator(dev_sentences1, dev_sentences2, dev_labels)


    def mining_evaluator(self):
        dev_sentences = {}
        dev_duplicates = []

        dataset_path = os.path.join(self.config['TRAIN']['DATASET_PATH'], self.config['TRAIN']['DATASET_MINING_CORPUS_PATH'])
        with open(dataset_path, encoding='utf8') as fIn:
            reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
            for row in reader:
                dev_sentences[row['qid']] = row['question']

                if len(dev_sentences) >= self.config['TRAIN']['MAX_DEV_SAMPLES']:
                    break

        dataset_path = os.path.join(self.config['TRAIN']['DATASET_PATH'], self.config['TRAIN']['DATASET_MINING_DUPLICATES_PATH'])
        with open(dataset_path, encoding='utf8') as fIn:
            reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
            for row in reader:
                if row['qid1'] in dev_sentences and row['qid2'] in dev_sentences:
                    dev_duplicates.append([row['qid1'], row['qid2']])

        return ParaphraseMiningEvaluator(dev_sentences, dev_duplicates, name='dev')


    def information_retrieval_evaluator(self):
        ir_queries = {}             #Our queries (qid => question)
        ir_needed_qids = set()      #QIDs we need in the corpus
        ir_corpus = {}              #Our corpus (qid => question)
        ir_relevant_docs = {}       #Mapping of relevant documents for a given query (qid => set([relevant_question_ids])

        dataset_path = os.path.join(self.config['TRAIN']['DATASET_PATH'], self.config['TRAIN']['DATASET_INFORMATION_RETRIEVAL_QUERIES_PATH'])
        with open(dataset_path, encoding='utf8') as fIn:
            next(fIn) #Skip header
            for line in fIn:
                qid, query, duplicate_ids = line.strip().split('\t')
                duplicate_ids = duplicate_ids.split(',')
                ir_queries[qid] = query
                ir_relevant_docs[qid] = set(duplicate_ids)

                for qid in duplicate_ids:
                    ir_needed_qids.add(qid)

        # First get all needed relevant documents (i.e., we must ensure, that the relevant questions are actually in the corpus
        distraction_questions = {}
        dataset_path = os.path.join(self.config['TRAIN']['DATASET_PATH'], self.config['TRAIN']['DATASET_INFORMATION_RETRIEVAL_CORPUS_PATH'])
        with open(dataset_path, encoding='utf8') as fIn:
            next(fIn) #Skip header
            for line in fIn:
                qid, question = line.strip().split('\t')

                if qid in ir_needed_qids:
                    ir_corpus[qid] = question
                else:
                    distraction_questions[qid] = question

        # Now, also add some irrelevant questions to fill our corpus
        other_qid_list = list(distraction_questions.keys())
        random.shuffle(other_qid_list)

        for qid in other_qid_list[0:max(0, self.config['TRAIN']['MAX_CORPUS_SIZE']-len(ir_corpus))]:
            ir_corpus[qid] = distraction_questions[qid]

        return InformationRetrievalEvaluator(ir_queries, ir_corpus, ir_relevant_docs)


    def sequential_evaluators(self):
        evaluators = []

        if 'CLASSIFICATION' in self.config['TRAIN']['EVALUATORS']:
            evaluators.append(self.classification_evaluator())
        
        if 'MINING' in self.config['TRAIN']['EVALUATORS']:
            evaluators.append(self.mining_evaluator())
        
        if 'INFORMATION_RETRIEVAL' in self.config['TRAIN']['EVALUATORS']:
            evaluators.append(self.information_retrieval_evaluator())
        
        return SequentialEvaluator(evaluators, main_score_function=lambda scores: scores[-1])