import asyncio
import json
import re
import time
from typing import AsyncIterator, Iterator, List, Optional, Tuple, Any
import httpx
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel, SimpleChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.outputs import ChatGenerationChunk, ChatResult, ChatGeneration
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from app.api.schemas import ChatMessage, SourceCitation
from app.core.config import settings
from app.core.logging import logger
from app.services.vector_store import VectorStoreService, vector_store_service


class GroqChatModel(BaseChatModel):
    """
    High-performance native Groq Chat Model supporting any Groq / OpenAI-compatible model,
    with full token streaming support via httpx, automatic error decoding, and offline fallback.
    """
    model_name: str = "openai/gpt-oss-120b"
    api_key: str = ""
    base_url: str = "https://api.groq.com/openai/v1"
    temperature: float = 0.0
    max_tokens: int = 2048

    @property
    def _llm_type(self) -> str:
        return "groq-chat"

    def _convert_messages(self, messages: List[BaseMessage]) -> List[dict]:
        payload = []
        for m in messages:
            role = "user"
            if m.type in ["assistant", "ai"]:
                role = "assistant"
            elif m.type == "system":
                role = "system"
            payload.append({"role": role, "content": str(m.content)})
        return payload

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs: Any) -> ChatResult:
        if not self.api_key or self.api_key.startswith("gsk_your_groq_api_key"):
            logger.warning("Placeholder or empty GROQ_API_KEY detected; falling back to grounded offline generation.")
            fallback = SmartMockChatModel()
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=fallback._call(messages)))])

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model_name,
            "messages": self._convert_messages(messages),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False
        }
        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.post(f"{self.base_url}/chat/completions", headers=headers, json=data)
                if resp.status_code != 200:
                    err_msg = self._parse_error_response(resp)
                    logger.error("Groq API error: %s", err_msg)
                    return ChatResult(generations=[ChatGeneration(message=AIMessage(content=f"⚠️ {err_msg}"))])

                result = resp.json()
                content = result["choices"][0]["message"]["content"]
                return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])
        except Exception as e:
            logger.warning("Groq connection failed (%s); falling back to grounded offline generation.", e)
            fallback = SmartMockChatModel()
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=fallback._call(messages)))])

    def _stream(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs: Any) -> Iterator[ChatGenerationChunk]:
        if not self.api_key or self.api_key.startswith("gsk_your_groq_api_key"):
            fallback = SmartMockChatModel()
            for chunk in fallback._stream(messages):
                yield chunk
            return

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model_name,
            "messages": self._convert_messages(messages),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True
        }
        try:
            with httpx.Client(timeout=60.0) as client:
                with client.stream("POST", f"{self.base_url}/chat/completions", headers=headers, json=data) as stream:
                    if stream.status_code != 200:
                        raw_err = stream.read().decode(errors="replace")
                        err_text = self._format_error_text(stream.status_code, raw_err)
                        logger.error("Groq stream HTTP %s: %s", stream.status_code, err_text)
                        yield ChatGenerationChunk(message=AIMessageChunk(content=f"⚠️ {err_text}"))
                        return

                    for line in stream.iter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:].strip()
                            if data_str == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data_str)
                                delta = chunk["choices"][0]["delta"].get("content", "")
                                if delta:
                                    yield ChatGenerationChunk(message=AIMessageChunk(content=delta))
                            except Exception:
                                continue
        except Exception as e:
            logger.warning("Groq stream connection failed (%s); falling back to offline stream.", e)
            fallback = SmartMockChatModel()
            for chunk in fallback._stream(messages):
                yield chunk

    def _parse_error_response(self, resp: httpx.Response) -> str:
        try:
            err_json = resp.json()
            if "error" in err_json and "message" in err_json["error"]:
                return f"Groq Error ({resp.status_code}): {err_json['error']['message']}"
        except Exception:
            pass
        return f"Groq Error ({resp.status_code}): {resp.text}"

    def _format_error_text(self, status_code: int, raw_text: str) -> str:
        try:
            err_json = json.loads(raw_text)
            if "error" in err_json and "message" in err_json["error"]:
                msg = err_json["error"]["message"]
                if "model" in msg.lower() and "does not exist" in msg.lower():
                    return (
                        f"Groq Model Error: {msg}\n\n"
                        "💡 Tip: Model is not recognized or not enabled on your Groq tier. "
                        "Please update LLM_MODEL in .env to a universally available Groq model like 'llama-3.1-8b-instant' or 'llama3-8b-8192'."
                    )
                return f"Groq Error ({status_code}): {msg}"
        except Exception:
            pass
        return f"Groq Error ({status_code}): {raw_text}"


class SmartMockChatModel(SimpleChatModel):
    """
    Intelligent offline mock model that synthesizes realistic grounded answers
    from provided context documents without requiring any external API key.
    Adheres strictly to anti-hallucination rules.
    """

    @property
    def _llm_type(self) -> str:
        return "smart-mock-chat"

    def _generate_grounded_answer(self, messages: List[BaseMessage]) -> str:
        user_query = ""
        context = ""
        for m in messages:
            if isinstance(m, HumanMessage):
                user_query = m.content
            elif isinstance(m, SystemMessage) and "Context:" in str(m.content):
                context = str(m.content).replace("Context:", "").strip()

        # Strip metadata header lines like [Source: ... ]
        clean_lines = [
            line.strip() for line in context.splitlines()
            if not line.strip().startswith("[Source:") and not line.strip().startswith("---")
        ]
        clean_context = " ".join(clean_lines).strip()

        if not clean_context or "No relevant context found" in clean_context:
            return (
                "Based on the provided documents, I do not have enough information "
                f"to answer the question: '{user_query}'."
            )

        # Extract keywords and sentences
        keywords = set(re.findall(r"\b\w{3,}\b", user_query.lower()))
        sentences = re.split(r"(?<=[.!?])\s+", clean_context)

        def word_match(w1: str, w2: str) -> bool:
            return w1 == w2 or (len(w1) >= 4 and len(w2) >= 4 and (w1.startswith(w2[:4]) or w2.startswith(w1[:4])))

        relevant_sentences = []
        for s in sentences:
            s_clean = s.strip()
            if not s_clean:
                continue
            s_words = set(re.findall(r"\b\w{3,}\b", s_clean.lower()))
            overlap_count = sum(1 for kw in keywords if any(word_match(kw, sw) for sw in s_words))
            if overlap_count > 0:
                relevant_sentences.append((s_clean, overlap_count))

        relevant_sentences.sort(key=lambda x: x[1], reverse=True)
        top_sentences = [s[0] for s in relevant_sentences[:4]]

        if top_sentences:
            synthesis = " ".join(top_sentences)
            return (
                f"Based on the referenced documentation:\n\n{synthesis}\n\n"
                "(Answer generated by grounded RAG pipeline)."
            )
        else:
            return (
                "Based on the provided documents, I do not have enough information "
                f"to answer the question: '{user_query}'."
            )

    def _call(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs) -> str:
        return self._generate_grounded_answer(messages)

    def _stream(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs):
        full_answer = self._generate_grounded_answer(messages)
        words = full_answer.split(" ")
        for i, word in enumerate(words):
            token = word if i == len(words) - 1 else word + " "
            yield ChatGenerationChunk(message=AIMessageChunk(content=token))


class RAGEngine:
    """
    Modern LCEL-powered Retrieval-Augmented Generation Engine.
    Features:
    - Multi-turn Conversational Memory / Contextual Query Reformulation
    - Anti-Hallucination Prompt Architecture
    - Source Citation and Similarity Attribution
    - Real-time Token Streaming
    - Dual Synchronous & Asynchronous Execution Interfaces
    """

    def __init__(self, vector_store: Optional[VectorStoreService] = None):
        self.vector_store = vector_store or vector_store_service
        self.llm = self._init_llm()

    def _init_llm(self) -> BaseChatModel:
        provider = settings.LLM_PROVIDER.lower()
        logger.info("Initializing LLM provider: %s (model: %s)", provider, settings.LLM_MODEL)

        if provider == "groq":
            try:
                from langchain_groq import ChatGroq
                logger.info("Using langchain_groq.ChatGroq with model: %s", settings.LLM_MODEL)
                return ChatGroq(
                    model_name=settings.LLM_MODEL,
                    groq_api_key=settings.GROQ_API_KEY,
                    temperature=settings.LLM_TEMPERATURE,
                    max_tokens=settings.LLM_MAX_TOKENS,
                )
            except Exception:
                try:
                    from langchain_openai import ChatOpenAI
                    logger.info("Using ChatOpenAI with Groq base URL for model: %s", settings.LLM_MODEL)
                    return ChatOpenAI(
                        model=settings.LLM_MODEL,
                        base_url=settings.GROQ_BASE_URL,
                        api_key=settings.GROQ_API_KEY,
                        temperature=settings.LLM_TEMPERATURE,
                        max_tokens=settings.LLM_MAX_TOKENS,
                    )
                except Exception:
                    logger.info("Using native GroqChatModel with model: %s", settings.LLM_MODEL)
                    return GroqChatModel(
                        model_name=settings.LLM_MODEL,
                        api_key=settings.GROQ_API_KEY or "",
                        base_url=settings.GROQ_BASE_URL,
                        temperature=settings.LLM_TEMPERATURE,
                        max_tokens=settings.LLM_MAX_TOKENS,
                    )

        elif provider == "openai":
            try:
                from langchain_openai import ChatOpenAI
                return ChatOpenAI(
                    model=settings.LLM_MODEL,
                    temperature=settings.LLM_TEMPERATURE,
                    max_tokens=settings.LLM_MAX_TOKENS,
                    api_key=settings.OPENAI_API_KEY
                )
            except Exception as e:
                logger.warning("Failed to initialize ChatOpenAI (%s). Falling back to SmartMockChatModel.", e)
                return SmartMockChatModel()

        elif provider == "google":
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                return ChatGoogleGenerativeAI(
                    model=settings.LLM_MODEL or "gemini-1.5-flash",
                    temperature=settings.LLM_TEMPERATURE,
                    google_api_key=settings.GOOGLE_API_KEY
                )
            except Exception as e:
                logger.warning("Failed to initialize Google LLM (%s). Falling back to SmartMockChatModel.", e)
                return SmartMockChatModel()

        elif provider == "ollama":
            try:
                from langchain_ollama import ChatOllama
                return ChatOllama(
                    model=settings.LLM_MODEL or "llama3.2",
                    base_url=settings.OLLAMA_BASE_URL,
                    temperature=settings.LLM_TEMPERATURE
                )
            except Exception as e:
                logger.warning("Failed to initialize ChatOllama (%s). Falling back to SmartMockChatModel.", e)
                return SmartMockChatModel()

        else:
            return SmartMockChatModel()

    def _convert_chat_history(self, history: Optional[List[ChatMessage]]) -> List[BaseMessage]:
        if not history:
            return []

        messages = []
        for msg in history:
            if msg.role == "user":
                messages.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                messages.append(AIMessage(content=msg.content))
            elif msg.role == "system":
                messages.append(SystemMessage(content=msg.content))
        return messages

    def reformulate_query_if_conversational(
        self,
        query: str,
        chat_history: List[BaseMessage]
    ) -> str:
        if not chat_history:
            return query

        reformulate_prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "Given a chat history and the latest user query which might reference context from earlier, "
                "formulate a standalone question that can be understood without the chat history. "
                "Do NOT answer the question, just reformulate it if needed, otherwise return it as is."
            )),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])

        chain = reformulate_prompt | self.llm | StrOutputParser()
        try:
            standalone_query = chain.invoke({"chat_history": chat_history, "input": query})
            logger.info("Reformulated conversational query: '%s' -> '%s'", query, standalone_query.strip())
            return standalone_query.strip()
        except Exception as e:
            logger.warning("Query reformulation failed (%s), using raw query.", e)
            return query

    def format_context_documents(self, docs_with_scores: List[Tuple[Document, float]]) -> str:
        if not docs_with_scores:
            return "No relevant context found in vector store."

        formatted_chunks = []
        for doc, score in docs_with_scores:
            source = doc.metadata.get("source", "Unknown")
            chunk_id = doc.metadata.get("chunk_id", 0)
            formatted_chunks.append(
                f"[Source: {source} | Chunk: {chunk_id} | Similarity: {score:.3f}]\n{doc.page_content}"
            )
        return "\n\n---\n\n".join(formatted_chunks)

    def extract_citations(self, docs_with_scores: List[Tuple[Document, float]]) -> List[SourceCitation]:
        citations = []
        for doc, score in docs_with_scores:
            snippet = doc.page_content.strip()
            if len(snippet) > 280:
                snippet = snippet[:280] + "..."
            citations.append(
                SourceCitation(
                    source=doc.metadata.get("source", "Unknown"),
                    chunk_id=doc.metadata.get("chunk_id"),
                    content_snippet=snippet,
                    similarity_score=round(score, 4),
                    metadata={k: v for k, v in doc.metadata.items() if k not in ["source", "chunk_id"]}
                )
            )
        return citations

    def query(
        self,
        query: str,
        chat_history: Optional[List[ChatMessage]] = None,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None
    ) -> Tuple[str, List[SourceCitation], float]:
        start_time = time.perf_counter()
        langchain_history = self._convert_chat_history(chat_history)

        effective_query = self.reformulate_query_if_conversational(query, langchain_history)

        docs_with_scores = self.vector_store.similarity_search_with_score(
            query=effective_query,
            k=top_k,
            threshold=similarity_threshold
        )
        context_text = self.format_context_documents(docs_with_scores)

        prompt = ChatPromptTemplate.from_messages([
            ("system", settings.SYSTEM_PROMPT),
            ("system", "Context:\n{context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])

        chain = prompt | self.llm | StrOutputParser()

        try:
            answer = chain.invoke({
                "context": context_text,
                "chat_history": langchain_history,
                "input": query
            })
        except Exception as e:
            logger.error("RAG generation error: %s", e)
            answer = f"Error generating answer: {str(e)}"

        execution_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
        citations = self.extract_citations(docs_with_scores)

        return answer, citations, execution_time_ms

    async def aquery(
        self,
        query: str,
        chat_history: Optional[List[ChatMessage]] = None,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None
    ) -> Tuple[str, List[SourceCitation], float]:
        return self.query(
            query=query,
            chat_history=chat_history,
            top_k=top_k,
            similarity_threshold=similarity_threshold
        )

    def stream_query(
        self,
        query: str,
        chat_history: Optional[List[ChatMessage]] = None,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None
    ):
        start_time = time.perf_counter()
        langchain_history = self._convert_chat_history(chat_history)

        effective_query = self.reformulate_query_if_conversational(query, langchain_history)

        docs_with_scores = self.vector_store.similarity_search_with_score(
            query=effective_query,
            k=top_k,
            threshold=similarity_threshold
        )
        context_text = self.format_context_documents(docs_with_scores)
        citations = self.extract_citations(docs_with_scores)

        yield {
            "event": "sources",
            "data": [c.model_dump() for c in citations]
        }

        prompt = ChatPromptTemplate.from_messages([
            ("system", settings.SYSTEM_PROMPT),
            ("system", "Context:\n{context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])

        chain = prompt | self.llm | StrOutputParser()

        for chunk in chain.stream({
            "context": context_text,
            "chat_history": langchain_history,
            "input": query
        }):
            yield {
                "event": "token",
                "data": chunk
            }

        execution_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
        yield {
            "event": "done",
            "data": {
                "execution_time_ms": execution_time_ms,
                "model_used": settings.LLM_MODEL
            }
        }


# Global singleton instance
rag_engine = RAGEngine()
