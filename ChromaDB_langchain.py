#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 19:16:06 2024

@author: jaykim
"""
from linkedIn_scaping import *

#from Assetlink_webscraping import * 

# !pip install chromadb openai langchain tiktoken

from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings

import dotenv
from langchain.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

import os
os.environ["OPENAI_API_KEY"] = "#####################################"

# load the document and split it into chunks
# 

dff.to_csv('/Users/jaykim/Downloads/temp.csv')
loader = CSVLoader("/Users/jaykim/Downloads/temp.csv")
documents = loader.load()

# split it into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)

all_splits = text_splitter.split_documents(documents)

#all_splits =  all_splits[0:3]

# create the open-source embedding function
embedding_function = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key='####################################')

REVIEWS_CHROMA_PATH = "/Users/jaykim/Desktop/chroma_data/"


# load it into Chroma
db = Chroma.from_documents(documents, embedding_function, persist_directory=REVIEWS_CHROMA_PATH)

# query it
query = "Who is Goerge"
docs = db.similarity_search(query)

dotenv.load_dotenv()

reviews_vector_db = Chroma(persist_directory=REVIEWS_CHROMA_PATH,
                           embedding_function=OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key='##############################'))

question = """What is his first name?"""

relevant_docs = reviews_vector_db.similarity_search(question, k=4)

# print results
print(docs[0].page_content)
print(docs)

data = pd.DataFrame(docs)

#import chromadb
#from chromadb.config import Settings
#client = chromadb.HttpClient(host='127.0.0.1', port=8000, settings=Settings(allow_reset=True, anonymized_telemetry=False))

#print(client.heartbeat())
#client = chromadb.HttpClient()







