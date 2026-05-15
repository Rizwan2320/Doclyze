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

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert technical research assistant.
Your task is to answer the user's question using ONLY the provided context.
Synthesize the information into a clear, concise, and professional answer.

CRITICAL RULES:
1. NEVER mention the word "Chunk" or include raw "[Chunk X | Page Y]" tags in your answer.
2. NEVER output raw mathematical notation, LaTeX, code fragments, or table markdown from the context unless the question explicitly asks for them.
3. If the context does not contain the answer, say EXACTLY: "I cannot answer this based on the provided document." Then STOP. Do not add any other text.
4. Do NOT use your general knowledge. If the answer is not in the Context section above, you MUST refuse.
5. Do not include introductory filler like "Based on the context..." or "According to the document..."
6. Answer with the specific information requested. Do not summarize unrelated sections.
"""),
            ("human", """Context:
{context}

Question: {question}

Answer using ONLY the Context above. If the answer is not in the Context, say "I cannot answer this based on the provided document." """),
        ])

    def format_docs(self, docs: List) -> str:
        """Format documents for the LLM context."""
        formatted = []
        for i, doc in enumerate(docs, 1):
            page = doc.metadata.get("page_number") or doc.metadata.get("page") or "N/A"
            content = doc.page_content.strip()
            formatted.append(f"[Chunk {i} | Page {page}]\n{content}")
        
        return "\n\n".join(formatted)

    def query(
        self, question: str, k: int = 15, collection_name: str = "default"
    ) -> Tuple[str, List]:
        """Query the RAG system and return (answer, retrieved_docs)."""
        docs = vector_store.similarity_search(
            question, k=k, collection_name=collection_name
        )

        logger.info(f"Retrieved {len(docs)} documents for query.")

        if not docs:
            return "I could not find the answer in the document.", []

        context = self.format_docs(docs)

        chain = self.prompt | self.llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": question})

        return answer.strip(), docs

    def summarize_document(self, collection_name: str = "default") -> str:
        """Summarize the documents in a collection."""
        logger.info(f"Summarizing collection: {collection_name}")

        docs = vector_store.similarity_search(
            "introduction overview abstract main topic key findings",
            k=15,
            collection_name=collection_name,
        )

        if not docs:
            return "No documents found in the collection."

        context = self.format_docs(docs)

        summary_prompt = ChatPromptTemplate.from_template(
            """Summarize the main topic and key points of the document in 3-4 concise sentences.
Use ONLY the provided context.

Context:
{context}

Summary:"""
        )

        chain = summary_prompt | self.llm | StrOutputParser()
        return chain.invoke({"context": context}).strip()


# Global singleton instance
rag_chain = RAGChain()