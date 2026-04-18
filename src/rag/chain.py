from typing import List, Tuple

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
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
            """You are a helpful research assistant. Answer the question based on the context below.
Use the information available in the context. If the context only partially addresses the question, answer based on what is there.
Only say you cannot answer if the context has absolutely no relevant information.

Context:
{context}

Question: {question}

Answer:"""
        )

    def format_docs(self, docs: List) -> str:
        """Format documents with chunk number and page info."""
        formatted = []
        for i, doc in enumerate(docs, 1):
            page = doc.metadata.get("page_number", "N/A")
            formatted.append(f"[Chunk {i} | Page {page}]\n{doc.page_content.strip()}")
        return "\n\n".join(formatted)

    def query(
        self, question: str, k: int = 15, collection_name: str = "default"
    ) -> Tuple[str, List]:
        """Query the RAG system and return answer + retrieved documents."""
        docs = vector_store.similarity_search(
            question, k=k, collection_name=collection_name
        )

        logger.info(f"Docs retrieved: {len(docs)}")
        for i, d in enumerate(docs[:2]):
            logger.info(f"Doc {i}: {repr(d.page_content[:100])}")

        if not docs:
            return "I could not find the answer in the document.", []

        context = self.format_docs(docs)

        chain = self.prompt | self.llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": question})

        return answer, docs

    def summarize_document(self, collection_name: str = "default") -> str:
        """Summarize the documents in a collection."""
        logger.info(f"Summarizing collection: {collection_name}")

        docs = vector_store.similarity_search(
            "introduction overview abstract main topic",
            k=15,
            collection_name=collection_name,
        )

        if not docs:
            return "No documents found."

        context = self.format_docs(docs)

        summary_prompt = ChatPromptTemplate.from_template(
            """Summarize the main topic and key points in 3-4 sentences.
Use ONLY the context below.

Context:
{context}

Summary:"""
        )

        chain = summary_prompt | self.llm | StrOutputParser()
        return chain.invoke({"context": context})


# Singleton / global instance
rag_chain = RAGChain()