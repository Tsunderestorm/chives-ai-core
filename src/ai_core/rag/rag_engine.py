"""RAG Engine for document retrieval and context enhancement."""

import asyncio
import logging
import os
import sys
from typing import List, Optional, Any

logger = logging.getLogger(__name__)


class RAGEngine:
    """Engine for retrieving relevant documents and enhancing context."""
    
    def __init__(self, collection_name: str, persist_directory: str, embedding_model: str, 
                 host: Optional[str] = None, port: Optional[int] = None):
        """Initialize the RAG engine.
        
        Args:
            collection_name: ChromaDB collection name
            persist_directory: Directory to persist ChromaDB data
            embedding_model: Sentence transformer model for embeddings
            host: ChromaDB server host (for client mode)
            port: ChromaDB server port (for client mode)
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model
        self.host = host
        self.port = port
        self.vector_store = None
        
    def _initialize_vector_store(self) -> bool:
        """Initialize the ChromaDB vector store connection."""
        try:
            logger.info(f"🔧 Initializing RAG retrieval")
            logger.info(f"🔧 Importing ChromaDBVectorStore")
            
            # Add chives-ingestion to Python path and import
            chives_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'chives-ingestion', 'src')
            if chives_path not in sys.path:
                sys.path.insert(0, chives_path)
            
            # Import the ChromaDB store implementation
            from chives_ingestion.implementations.vector_store.chromadb_store import ChromaDBVectorStore
            
            logger.info(f"🔧 Logging configuration values")
            # Log the configuration values being used
            logger.info(f"🔧 Initializing ChromaDBVectorStore with:")
            logger.info(f"   Collection Name: {self.collection_name}")
            logger.info(f"   Persist Directory: {self.persist_directory}")
            logger.info(f"   Embedding Model: {self.embedding_model}")
            logger.info(f"   Host: {self.host}")
            logger.info(f"   Port: {self.port}")
            
            # Check if the persist directory exists and log its absolute path
            if self.persist_directory:
                abs_path = os.path.abspath(self.persist_directory)
                logger.info(f"   Absolute Persist Path: {abs_path}")
                if os.path.exists(abs_path):
                    logger.info(f"   ✅ Persist directory exists")
                else:
                    logger.warning(f"   ⚠️  Persist directory does not exist: {abs_path}")
            else:
                logger.info("   Persist directory not specified (using default)")
            
            # Initialize vector store
            self.vector_store = ChromaDBVectorStore(
                collection_name=self.collection_name,
                persist_directory=self.persist_directory,
                embedding_model=self.embedding_model,
                host=self.host,
                port=self.port
            )
            
            logger.info(f"✅ RAGEngine initialized with ChromaDB collection: {self.collection_name}")
            return True
            
        except ImportError as e:
            logger.warning(f"Chives-ingestion not available for RAG: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            return False

    async def retrieve_relevant_documents_async(self, query: str, max_results: int = 3) -> Optional[str]:
        """Async version of document retrieval - this is the main method that should be used."""
        if not self.vector_store:
            if not self._initialize_vector_store():
                return None
        
        try:
            if not await self.vector_store.connect():
                logger.warning("Failed to connect to ChromaDB for RAG")
                return None
            
            # Perform semantic search
            # This is plagued with this error: https://github.com/chroma-core/chroma/issues/870#issuecomment-1898664798 
            # I've updated the DB in the pyproject but there might be more to do
            results = await self.vector_store.search_by_text(query, limit=max_results)
            
            if not results:
                return None
            
            # Format the retrieved documents for context
            formatted_context = self._format_rag_context(results, query)
            return formatted_context
            
        except Exception as e:
            logger.warning(f"RAG retrieval failed: {e}")
            return None
    
    def _format_rag_context(self, search_results: list, query: str) -> str:
        """Format search results into a context string for RAG enhancement.
        
        Args:
            search_results: List of search results from the vector store
            query: The original user query
            
        Returns:
            Formatted string with relevant document content
        """
        if not search_results:
            return ""
        
        context_parts = [f"Based on your query '{query}', here are relevant documents:"]
        
        for i, result in enumerate(search_results, 1):
            doc = result.document
            score = result.score
            
            # Extract metadata
            title = doc.metadata.get('document_title', 'Untitled Document')
            source = doc.metadata.get('document_source', 'Unknown Source')
            
            # Format the content (first 300 characters for context)
            content_preview = doc.content[:300]
            if len(doc.content) > 300:
                content_preview += "..."
            
            context_parts.append(
                f"\n{i}. {title} (Relevance: {score:.3f})\n"
                f"   Source: {source}\n"
                f"   Content: {content_preview}"
            )
        
        return "\n".join(context_parts)
    
    def is_available(self) -> bool:
        """Check if the RAG engine is available and properly configured."""
        if not self.vector_store:
            return self._initialize_vector_store()
        return True
