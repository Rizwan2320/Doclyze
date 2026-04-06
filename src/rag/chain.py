from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from loguru import logger

from src.config.settings import settings
from src.vectorstore.chroma import vector_store


class RAGChain:

    def __init__(self):
        self.llm = ChatGroq(
            model=settings.GROQ_MODEL,
            temperature=0.0,
            max_tokens=1500,
            api_key=settings.GROQ_API_KEY.get_secret_value(),
        )

        self.prompt = ChatPromptTemplate.from_template(
            """You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say "I could not find the answer in the document."
Do not use your own knowledge. Do not guess.

Context:
{context}

Question: {question}

Answer:"""
        )

    def format_docs(self, docs: List) -> str:
        formatted = []
        for i, doc in enumerate(docs, 1):
            page = doc.metadata.get("page_number", "N/A")
            formatted.append(f"[Chunk {i} | Page {page}]\n{doc.page_content.strip()}")
        return "\n\n".join(formatted)

    def query(self, question: str, k: int = 10) -> str:
        logger.info(f"Query: '{question}' | k={k}")

        docs = vector_store.similarity_search(question, k=k)

        if not docs:
            return "No documents found in the vector store."

        logger.info(f"Retrieved {len(docs)} chunks")
        context = self.format_docs(docs)

        chain = self.prompt | self.llm | StrOutputParser()
        return chain.invoke({"context": context, "question": question})

    def summarize_document(self) -> str:
        logger.info("Generating summary...")

        docs = vector_store.similarity_search(
            "introduction overview abstract main topic", k=15
        )

        if not docs:
            return "No documents found in the vector store."

        context = self.format_docs(docs)

        summary_prompt = ChatPromptTemplate.from_template(
            """Summarize the main topic and key points of this document in 3-4 sentences.
Use ONLY the context below. Do not use outside knowledge.

Context:
{context}

Summary:"""
        )

        chain = summary_prompt | self.llm | StrOutputParser()
        return chain.invoke({"context": context})


# Singleton
rag_chain = RAGChain()