

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

import os
from typing import Any, Dict, List
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores.pinecone import Pinecone as PineconeLangChain
from pinecone import Pinecone


#pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

pc = Pinecone(api_key='f7f4a832-58cf-4606-98b8-514fad1c6e86')

INDEX_NAME = "test"


def run_llm(query: str, chat_history: List[Dict[str, Any]] = []):
   # embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    embeddings = OpenAIEmbeddings(openai_api_key="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")
    docsearch = PineconeLangChain.from_existing_index(
        embedding=embeddings,
        index_name=INDEX_NAME,
    )
    chat = ChatOpenAI(
        verbose=True,
        temperature=0,
    )

    qa = ConversationalRetrievalChain.from_llm(
        llm=chat, retriever=docsearch.as_retriever(), return_source_documents=True
    )
    return qa.invoke({"question": query, "chat_history": chat_history})
