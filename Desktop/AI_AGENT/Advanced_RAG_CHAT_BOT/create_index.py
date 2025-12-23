import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Sample documents
documents = [
    Document(page_content="FAISS is a library for efficient similarity search and clustering of dense vectors. It contains algorithms that search in sets of vectors of any size, up to ones that possibly do not fit in RAM."),
    Document(page_content="Streamlit is an open-source Python library that makes it easy to create and share beautiful, custom web apps for machine learning and data science."),
    Document(page_content="LangGraph is a library for building stateful, multi-actor applications with LLMs, built on top of LangChain."),
    Document(page_content="Google Gemini is a multimodal large language model developed by Google DeepMind."),
    Document(page_content="Retrieval-Augmented Generation (RAG) combines the power of large language models with external knowledge retrieval for more accurate and up-to-date responses."),
    Document(page_content="Vector databases store and retrieve high-dimensional vectors efficiently, enabling fast similarity searches for applications like recommendation systems and semantic search."),
    Document(page_content="Embeddings are dense vector representations of text, images, or other data that capture semantic meaning and relationships."),
    Document(page_content="Machine learning models can be fine-tuned on specific datasets to improve performance on particular tasks."),
    Document(page_content="Natural Language Processing (NLP) involves teaching computers to understand, interpret, and generate human language."),
    Document(page_content="Artificial Intelligence is transforming industries by automating tasks, providing insights, and enabling new capabilities.")
]

# Create FAISS index
vectorstore = FAISS.from_documents(documents, embeddings)

# Save the index
vectorstore.save_local("faiss_index")

print("FAISS index created successfully with sample documents!")
