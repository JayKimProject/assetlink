#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 00:13:41 2024

@author: jaykim
"""

# create_retriever.py

import dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

REVIEWS_CHROMA_PATH = "/Users/jaykim/Desktop/chroma_data/"

embedding_function = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key='############################')

dotenv.load_dotenv()

reviews_vector_db = Chroma(
    persist_directory=REVIEWS_CHROMA_PATH,
     embedding_function=embedding_function, )

question = """who is George?"""
relevant_docs = reviews_vector_db.similarity_search(question, k=3)
relevant_docs[0].page_content
relevant_docs[1].page_content
relevant_docs[2].page_content

#answer = result['result']
#context = " ".join([d.page_content for d in result['source_documents']])

answer = """who is George?"""
context = " ".join([d.page_content for d in relevant_docs])

import torch
from sacrebleu import corpus_bleu
from rouge_score import rouge_scorer
from bert_score import score
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
import nltk
from nltk.util import ngrams

class RAGEvaluator:
    def __init__(self):
        self.gpt2_model, self.gpt2_tokenizer = self.load_gpt2_model()
        self.bias_pipeline = pipeline("zero-shot-classification", model="Hate-speech-CNERG/dehatebert-mono-english")
    def load_gpt2_model(self):
        model = GPT2LMHeadModel.from_pretrained('gpt2')
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        return model, tokenizer
    def evaluate_bleu_rouge(self, candidates, references):
        bleu_score = corpus_bleu(candidates, [references]).score
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge_scores = [scorer.score(ref, cand) for ref, cand in zip(references, candidates)]
        rouge1 = sum([score['rouge1'].fmeasure for score in rouge_scores]) / len(rouge_scores)
        return bleu_score, rouge1
    def evaluate_bert_score(self, candidates, references):
        P, R, F1 = score(candidates, references, lang="en", model_type='bert-base-multilingual-cased')
        return P.mean().item(), R.mean().item(), F1.mean().item()
    def evaluate_perplexity(self, text):
        encodings = self.gpt2_tokenizer(text, return_tensors='pt')
        max_length = self.gpt2_model.config.n_positions
        stride = 512
        lls = []
        for i in range(0, encodings.input_ids.size(1), stride):
            begin_loc = max(i + stride - max_length, 0)
            end_loc = min(i + stride, encodings.input_ids.size(1))
            trg_len = end_loc - i
            input_ids = encodings.input_ids[:, begin_loc:end_loc]
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100
            with torch.no_grad():
                outputs = self.gpt2_model(input_ids, labels=target_ids)
                log_likelihood = outputs[0] * trg_len
            lls.append(log_likelihood)
        ppl = torch.exp(torch.stack(lls).sum() / end_loc)
        return ppl.item()
    def evaluate_diversity(self, texts):
        all_tokens = [tok for text in texts for tok in text.split()]
        unique_bigrams = set(ngrams(all_tokens, 2))
        diversity_score = len(unique_bigrams) / len(all_tokens) if all_tokens else 0
        return diversity_score
    def evaluate_racial_bias(self, text):
        results = self.bias_pipeline([text], candidate_labels=["hate speech", "not hate speech"])
        bias_score = results[0]['scores'][results[0]['labels'].index('hate speech')]
        return bias_score
    def evaluate_all(self, question, response, reference):
        candidates = [response]
        references = [reference]
        bleu, rouge1 = self.evaluate_bleu_rouge(candidates, references)
        bert_p, bert_r, bert_f1 = self.evaluate_bert_score(candidates, references)
        perplexity = self.evaluate_perplexity(response)
        diversity = self.evaluate_diversity(candidates)
        racial_bias = self.evaluate_racial_bias(response)
        return {
            "BLEU": bleu,
            "ROUGE-1": rouge1,
            "BERT P": bert_p,
            "BERT R": bert_r,
            "BERT F1": bert_f1,
            "Perplexity": perplexity,
            "Diversity": diversity,
            "Racial Bias": racial_bias
        }

evaluator = RAGEvaluator()
question = "What is mentioned about Composability?"
response = answer 
reference = context
metrics = evaluator.evaluate_all(question, response, reference)

for k,v in metrics.items():
    if k == 'BLEU':
        print(f"BLEU measures the overlap between the generated output and reference text based on n-grams. Higher scores indicate better match.score obtained :{v}")
    elif k == "ROUGE-1":
        print(f"ROUGE-1 measures the overlap of unigrams between the generated output and reference text. Higher scores indicate better match.Score obtained:{v}") 
    elif k == 'BERT P':
        print(f"BERTScore evaluates the semantic similarity between the generated output and reference text using BERT embeddings.")
        print(f"\n\n**BERT Precision**: {metrics['BERT P']}")
        print(f"**BERT Recall**: {metrics['BERT R']}")
        print(f"**BERT F1 Score**: {metrics['BERT F1']}")
    elif k == 'Perplexity':
        print(f"Perplexity measures how well a language model predicts the text. Lower values indicate better fluency and coherence. score obtained :{v}")
    elif k == 'Diversity':
        print(f"Diversity measures the uniqueness of bigrams in the generated output. Higher values indicate more diverse and varied output.score obtained:{v}")
    elif k == 'Racial Bias':
        print(f"Racial Bias score indicates the presence of biased language in the generated output. Higher scores indicate more bias.score obtained:{v}")



