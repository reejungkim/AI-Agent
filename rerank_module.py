"""
Reranking Module for RAG Systems
================================

This module provides reranking functionality using cross-encoder models
to improve retrieval accuracy in RAG (Retrieval-Augmented Generation) systems.

Based on the user's memory preferences: all-MiniLM-L6-v2 and llama3-8b-8192 for speed.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import logging

try:
    from sentence_transformers import CrossEncoder
    RERANK_AVAILABLE = True
except ImportError:
    RERANK_AVAILABLE = False
    logging.warning("sentence-transformers not available. Install with: pip install sentence-transformers")

from langchain.schema import Document


@dataclass
class RerankConfig:
    """Configuration for reranking."""
    model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    threshold: float = 0.0
    max_length: int = 512
    batch_size: int = 32


class DocumentReranker:
    """
    Document reranker using cross-encoder models.
    
    This class implements a two-stage retrieval approach:
    1. Initial retrieval using embedding similarity
    2. Reranking using cross-encoder models for better accuracy
    """
    
    def __init__(self, config: RerankConfig = None):
        """
        Initialize the reranker.
        
        Args:
            config: Reranking configuration
        """
        self.config = config or RerankConfig()
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the cross-encoder model."""
        if not RERANK_AVAILABLE:
            logging.error("sentence-transformers not available")
            return
        
        try:
            self.model = CrossEncoder(
                self.config.model_name,
                max_length=self.config.max_length
            )
            logging.info(f"Loaded reranker model: {self.config.model_name}")
        except Exception as e:
            logging.error(f"Failed to load reranker model: {e}")
            self.model = None
    
    def is_available(self) -> bool:
        """Check if reranking is available."""
        return self.model is not None
    
    def rerank_documents(
        self, 
        query: str, 
        documents: List[Document], 
        top_k: Optional[int] = None
    ) -> List[Tuple[Document, float]]:
        """
        Rerank documents based on query relevance.
        
        Args:
            query: Query string
            documents: List of documents to rerank
            top_k: Number of top documents to return (if None, returns all)
        
        Returns:
            List of (document, score) tuples sorted by relevance score
        """
        if not self.is_available():
            logging.warning("Reranker not available, returning original order")
            return [(doc, 1.0) for doc in documents]
        
        if not documents:
            return []
        
        # Prepare query-document pairs
        pairs = [(query, doc.page_content) for doc in documents]
        
        # Get scores
        try:
            scores = self.model.predict(pairs)
            
            # Combine documents with scores
            doc_scores = list(zip(documents, scores))
            
            # Filter by threshold
            filtered_docs = [
                (doc, float(score)) for doc, score in doc_scores 
                if score >= self.config.threshold
            ]
            
            # Sort by score (descending)
            sorted_docs = sorted(filtered_docs, key=lambda x: x[1], reverse=True)
            
            # Apply top_k limit
            if top_k is not None:
                sorted_docs = sorted_docs[:top_k]
            
            logging.info(f"Reranked {len(documents)} documents, returned {len(sorted_docs)}")
            return sorted_docs
            
        except Exception as e:
            logging.error(f"Reranking failed: {e}")
            return [(doc, 1.0) for doc in documents]
    
    def rerank_documents_simple(
        self, 
        query: str, 
        documents: List[Document], 
        top_k: Optional[int] = None
    ) -> List[Document]:
        """
        Simplified reranking that returns only documents.
        
        Args:
            query: Query string
            documents: List of documents to rerank
            top_k: Number of top documents to return
        
        Returns:
            List of reranked documents
        """
        ranked_docs_scores = self.rerank_documents(query, documents, top_k)
        
        # Add reranking metadata to documents
        ranked_docs = []
        for i, (doc, score) in enumerate(ranked_docs_scores):
            # Create a copy to avoid modifying original
            new_doc = Document(
                page_content=doc.page_content,
                metadata=doc.metadata.copy() if doc.metadata else {}
            )
            new_doc.metadata.update({
                'rerank_score': score,
                'rerank_position': i + 1,
                'original_position': documents.index(doc) + 1
            })
            ranked_docs.append(new_doc)
        
        return ranked_docs


class RetrievalWithReranking:
    """
    Enhanced retrieval class that combines vector search with reranking.
    
    This class can be used as a drop-in replacement for standard retrievers
    in LangChain applications.
    """
    
    def __init__(
        self, 
        vectorstore, 
        reranker: DocumentReranker = None,
        k_initial: int = 10,
        k_final: int = 4
    ):
        """
        Initialize retrieval with reranking.
        
        Args:
            vectorstore: Vector store for initial retrieval
            reranker: Document reranker instance
            k_initial: Number of documents for initial retrieval
            k_final: Number of documents after reranking
        """
        self.vectorstore = vectorstore
        self.reranker = reranker or DocumentReranker()
        self.k_initial = k_initial
        self.k_final = k_final
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """
        Retrieve and rerank documents.
        
        Args:
            query: Query string
        
        Returns:
            List of relevant documents
        """
        # Initial retrieval
        initial_docs = self.vectorstore.similarity_search(query, k=self.k_initial)
        
        if not self.reranker.is_available():
            return initial_docs[:self.k_final]
        
        # Rerank documents
        reranked_docs = self.reranker.rerank_documents_simple(
            query, initial_docs, self.k_final
        )
        
        return reranked_docs
    
    async def aget_relevant_documents(self, query: str) -> List[Document]:
        """Async version (calls sync method)."""
        return self.get_relevant_documents(query)


def create_reranking_qa_chain(
    vectorstore,
    llm,
    prompt_template=None,
    reranker_config: RerankConfig = None,
    k_initial: int = 10,
    k_final: int = 4
):
    """
    Create a QA chain with reranking capability.
    
    Args:
        vectorstore: Vector store for retrieval
        llm: Language model
        prompt_template: Prompt template (optional)
        reranker_config: Reranker configuration
        k_initial: Initial retrieval count
        k_final: Final document count after reranking
    
    Returns:
        QA chain with reranking
    """
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate
    
    # Default prompt template
    if prompt_template is None:
        prompt_template = """Use the following context to answer the question accurately and concisely.

Context: {context}

Question: {question}

Answer:"""
    
    # Create prompt
    if isinstance(prompt_template, str):
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
    else:
        prompt = prompt_template
    
    # Create reranker
    reranker = DocumentReranker(reranker_config)
    
    # Create retriever with reranking
    retriever = RetrievalWithReranking(
        vectorstore=vectorstore,
        reranker=reranker,
        k_initial=k_initial,
        k_final=k_final
    )
    
    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    
    return qa_chain


# Available reranker models
RERANKER_MODELS = {
    "ms-marco-miniLM-L6-v2": {
        "model": "cross-encoder/ms-marco-MiniLM-L6-v2",
        "description": "Fast and balanced model for general reranking",
        "speed": "fast",
        "accuracy": "good"
    },
    "ms-marco-miniLM-L12-v2": {
        "model": "cross-encoder/ms-marco-MiniLM-L12-v2", 
        "description": "Better accuracy with moderate speed",
        "speed": "medium",
        "accuracy": "very good"
    },
    "ms-marco-TinyBERT-L6": {
        "model": "cross-encoder/ms-marco-TinyBERT-L-6",
        "description": "Fastest model with decent accuracy",
        "speed": "very fast", 
        "accuracy": "good"
    },
    "qnli-electra-base": {
        "model": "cross-encoder/qnli-electra-base",
        "description": "Specialized for question-answering tasks",
        "speed": "medium",
        "accuracy": "very good"
    }
}


def get_reranker_config(model_name: str, threshold: float = 0.0) -> RerankConfig:
    """
    Get reranker configuration for a specific model.
    
    Args:
        model_name: Name of the reranker model
        threshold: Score threshold for filtering
    
    Returns:
        RerankConfig instance
    """
    if model_name in RERANKER_MODELS:
        return RerankConfig(
            model_name=RERANKER_MODELS[model_name]["model"],
            threshold=threshold
        )
    else:
        # Assume it's a direct model path
        return RerankConfig(
            model_name=model_name,
            threshold=threshold
        )


# Example usage and helper functions
def example_usage():
    """Example of how to use the reranking module."""
    
    print("=== Reranking Module Example ===")
    
    # 1. Create reranker configuration (using user's preferred fast model)
    config = get_reranker_config("ms-marco-miniLM-L6-v2", threshold=0.0)
    
    # 2. Create reranker
    reranker = DocumentReranker(config)
    
    if not reranker.is_available():
        print("Reranker not available. Install sentence-transformers first.")
        return
    
    # 3. Example documents
    docs = [
        Document(page_content="Python is a programming language.", metadata={"source": "doc1"}),
        Document(page_content="Machine learning uses algorithms.", metadata={"source": "doc2"}),
        Document(page_content="Python is great for machine learning.", metadata={"source": "doc3"}),
    ]
    
    # 4. Rerank documents
    query = "What is Python used for?"
    reranked = reranker.rerank_documents_simple(query, docs, top_k=2)
    
    print(f"Query: {query}")
    print("Reranked documents:")
    for i, doc in enumerate(reranked):
        score = doc.metadata.get('rerank_score', 'N/A')
        print(f"{i+1}. {doc.page_content} (score: {score})")


if __name__ == "__main__":
    example_usage()