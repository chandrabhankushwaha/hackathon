#from dotenv import load_dotenv

#load_dotenv()

import os

from langchain_community.document_loaders import ReadTheDocsLoader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain_community.vectorstores import pinecone as PineconeLangChain
from langchain_pinecone import PineconeVectorStore
#from langchain_community.vectorstores import Pinecone
#from langchain_community.vectorstores import pinecone
from pinecone import Pinecone
from langchain_community.document_loaders import PyPDFLoader

pc = Pinecone(
   # api_key=os.environ.get("PINECONE_API_KEY"),
    #environment=os.environ["PINECONE_ENVIRONMENT_REGION"]
    api_key="f7f4a832-58cf-4606-98b8-514fad1c6e86",
    environment="gcp-starter",
)

INDEX_NAME = "test"

pdf_path='C:\\Users\\Shilpi\\Desktop\\Hack\\code-of-conduct.pdf'
   

def ingest_docs():
    loader=PyPDFLoader(file_path=pdf_path)
    raw_documents = loader.load()
    print(f"loaded {len(raw_documents)} documents")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    documents = text_splitter.split_documents(raw_documents)
    for doc in documents:
        new_url = doc.metadata["source"]
        new_url = new_url.replace("egs-docs", "https:/")
        doc.metadata.update({"source": new_url})

    #embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
       # embeddings = OpenAIEmbeddings(openai_api_key="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")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small",openai_api_key="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")
    print(f"Going to add {len(documents)} to Pinecone")
   # PineconeLangChain.from_documents(documents, embeddings, index_name="langchain-doc-index")

    PineconeVectorStore.from_documents(documents, embeddings, index_name="langchain-doc-index")
    
    print("****Loading to vectorstore done ***")


if __name__ == "__main__":
    ingest_docs()
