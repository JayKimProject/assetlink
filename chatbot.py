#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 00:17:07 2024

@author: jaykim
"""
import dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import (
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)

review_template_str = """Your job is to use 
LinkedIn and BrokerCheck to answer questions about their information. 
Use the following context to answer questions.
Be as detailed as possible, but don't make up any information
that's not from the context. If you don't know an answer, say
you don't know.
{context}
"""
review_system_prompt = SystemMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["context"],
        template=review_template_str,
    )
)
review_human_prompt = HumanMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["question"],
        template="{question}",
    )
)
messages = [review_system_prompt, review_human_prompt]

review_prompt_template = ChatPromptTemplate(
    input_variables=["context", "question"],
    messages=messages,
)

chat_model = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0, openai_api_key='###################################')
review_chain = review_prompt_template | chat_model

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

REVIEWS_CHROMA_PATH = "/Users/jaykim/Desktop/chroma_data/"
embedding_function = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key='####################################')


reviews_vector_db = Chroma(
    persist_directory=REVIEWS_CHROMA_PATH,
    embedding_function=embedding_function
)

reviews_retriever  = reviews_vector_db.as_retriever(k=10)

import dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import (
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)
from langchain_core.output_parsers import StrOutputParser


output_parser = StrOutputParser()

review_chain = review_prompt_template | chat_model | output_parser


### Ask some questions to chatbot. @@@@@
########################################


context = "his name is George"
question = "explain who George is?"

review_chain.invoke({"context": context, "question": question})
#AIMessage(content='Yes, the patient had a great stay and had a
#positive experience at the hospital.')


