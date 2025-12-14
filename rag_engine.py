import os
import shutil
from typing import List, Optional, Dict, Any
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage

# Configuration
PERSIST_DIRECTORY = "db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL = "llama3"

class RagEngine:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        self.vector_store = None
        self._init_vector_store()

    def _init_vector_store(self):
        """Initializes the ChromaDB vector store."""
        if os.path.exists(PERSIST_DIRECTORY):
            self.vector_store = Chroma(
                persist_directory=PERSIST_DIRECTORY,
                embedding_function=self.embeddings
            )
        else:
            self.vector_store = None 

    def ingest_pdf(self, file_path: str, original_filename: str = None) -> str:
        """
        Loads a PDF, splits it, and adds it to the vector store.
        Args:
            file_path: The path to the temporary file on disk.
            original_filename: The actual name of the file uploaded by the user.
        """
        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
            # OVERWRITE METADATA: Use original filename instead of temp path
            for doc in documents:
                if original_filename:
                    doc.metadata["source"] = original_filename
            
            # Ensure safe split
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = text_splitter.split_documents(documents)
            
            if self.vector_store is None:
                self.vector_store = Chroma.from_documents(
                    documents=chunks,
                    embedding=self.embeddings,
                    persist_directory=PERSIST_DIRECTORY
                )
            else:
                self.vector_store.add_documents(chunks)
                
            return f"Procesado correctamente: {len(chunks)} fragmentos de '{original_filename}'."
        except Exception as e:
            return f"Error al procesar PDF: {str(e)}"

    def get_ingested_files(self) -> List[str]:
        """Returns a list of unique filenames currently in the database."""
        if not self.vector_store:
            return []
        
        try:
            # ChromaDB get() returns specific metadata
            data = self.vector_store.get()
            if not data or 'metadatas' not in data:
                return []
            
            unique_files = set()
            for meta in data['metadatas']:
                if meta and 'source' in meta:
                    unique_files.add(os.path.basename(meta['source']))
            
            return list(unique_files)
        except Exception:
            return []

    def delete_file(self, filename: str) -> str:
        """Deletes all chunks associated with a specific filename."""
        if not self.vector_store:
            return "Error: Base de datos no inicializada."
            
        try:
            # Delete where metadata 'source' matches filename
            # Note: Chroma expects a filter dict
            self.vector_store.delete(where={"source": filename})
            return f"Archivo '{filename}' eliminado de la memoria."
        except Exception as e:
            return f"Error al eliminar archivo: {str(e)}"

    def clear_database(self):
        """(Deprecated) Clears the existing vector database."""
        # User requested to hide this, keeping it for internal utility just in case
        if os.path.exists(PERSIST_DIRECTORY):
            try:
                self.vector_store = None
                shutil.rmtree(PERSIST_DIRECTORY)
                return "Base de datos borrada correctamente."
            except Exception as e:
                return f"Error al borrar base de datos: {e}"
        return "La base de datos ya estaba vacía."
            
    def get_chain(self):
        """Returns a Conversational RAG chain using LCEL."""
        if not self.vector_store:
            return None

        llm = Ollama(model=LLM_MODEL)
        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )

        # 1. Contextualize Question Chain
        contextualize_q_system_prompt = """Dado un historial de chat y la última pregunta del usuario 
        (que podría hacer referencia al contexto del historial), formula una pregunta independiente 
        que pueda entenderse sin el historial. NO la respondas, solo re-formúlala si es necesario 
        o devuélvela tal cual si ya es clara."""
        
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])
        
        history_aware_retriever = (
            contextualize_q_prompt
            | llm
            | StrOutputParser()
            | retriever
        )

        # 2. Answer Chain
        qa_system_prompt = """Eres un asistente experto para tareas de preguntas y respuestas. 
        Usa los siguientes fragmentos de contexto recuperado para responder a la pregunta. 
        Si no sabes la respuesta, di simplemente que no lo sabes. 
        Usa un máximo de tres oraciones y mantén la respuesta concisa.
        RESPONDE SIEMPRE EN ESPAÑOL.
        
        {context}"""
        
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # FIX: Explicitly wrap python function in RunnableLambda to allow piping
        rag_chain = (
            RunnablePassthrough.assign(
                context=(history_aware_retriever | RunnableLambda(format_docs))
            )
            | qa_prompt
            | llm
            | StrOutputParser()
        )

        return rag_chain
