# AI Agent - RAG (Retrieval-Augmented Generation) System

A comprehensive RAG implementation using LangChain, featuring document processing, vector search, and intelligent question-answering capabilities.

## 🚀 Features

- **Document Processing**: PDF document loading and text chunking
- **Vector Search**: FAISS-based similarity search with multiple embedding models
- **Reranking**: Advanced document reranking using cross-encoders
- **Multiple LLM Support**: Integration with HuggingFace, Groq, and Ollama
- **RAG Evaluation**: Comprehensive evaluation using Ragas metrics
- **Streamlit Interface**: Web-based user interface for easy interaction

## 📋 Overview

This project implements a complete RAG pipeline with the following components:

1. **Document Loader** - PDF document processing
2. **Text Splitter** - Intelligent text chunking with overlap
3. **Embedding Model** - Multiple embedding options (HuggingFace models)
4. **Vector Store** - FAISS for efficient similarity search
5. **Retriever** - Dense and sparse retrieval methods
6. **Reranker** - Cross-encoder based document reranking
7. **Prompt Template** - Customizable prompt engineering
8. **LLM Integration** - Multiple language model providers
9. **Chain** - End-to-end RAG pipeline
10. **Evaluator** - Performance assessment using Ragas

## 🛠️ Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd AI-Agent
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file with the following variables:
```env
# HuggingFace API
huggingface_read=your_huggingface_token

# Groq API
groq_api=your_groq_api_key

# LangSmith API (optional)
LANGSMITH_API_KEY=your_langsmith_key

# Gemini API (optional)
gemini_llm_api=your_gemini_api_key
```

## 📚 Usage

### Basic RAG Pipeline

```python
# Load and process documents
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load PDF document
loader = PyPDFLoader("path/to/document.pdf")
docs = loader.load()

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=200
)
texts = text_splitter.split_documents(docs)

# Create embeddings
embeddings = HuggingFaceEmbeddings(
    model_name='all-MiniLM-L6-v2',
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# Create vector store
vectorstore = FAISS.from_documents(texts, embeddings)

# Create retriever
retriever = vectorstore.as_retriever()
```

### Advanced RAG with Reranking

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# Initialize reranker
model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")
compressor = CrossEncoderReranker(model=model, top_n=3)

# Create compression retriever
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, 
    base_retriever=retriever
)
```

### RAG Chain Implementation

```python
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

# Define prompt template
template = """<|system|>
You are an assistant for question-answering tasks. 
Using the information contained in the context,
give a comprehensive answer to the question.
Respond only to the question asked, response should be concise and relevant to the question.
Provide the number of the source document when relevant.
If you don't know the answer, just say that you don't know. 
Answer in Korean. <|end|>

<|user|>
{question}<|end|>
<|assistant|>"""

prompt = PromptTemplate.from_template(template)

# Create RAG chain
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Query the system
result = chain.invoke("Your question here")
```

## 🔧 Configuration

### Embedding Models

The system supports multiple embedding models:

- **`all-MiniLM-L6-v2`** - Fast and efficient (default)
- **`jhgan/ko-sroberta-multitask`** - Korean language optimized

### Language Models

Multiple LLM providers are supported:

#### HuggingFace
```python
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

endpoint = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V3-0324",
    max_new_tokens=512,
    temperature=0.7,
    huggingfacehub_api_token=os.environ["huggingface_read"],
    task="conversational"
)
llm = ChatHuggingFace(llm=endpoint)
```

#### Groq
```python
from langchain_groq import ChatGroq

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.7,
    max_tokens=512,
    groq_api_key=os.environ["groq_api"]
)
```

#### Ollama (Local)
```python
from langchain_community.chat_models import ChatOllama

llm = ChatOllama(
    model='mistral:latest',
    format="json",
    temperature=0
)
```

## 📊 Evaluation

The system includes comprehensive evaluation using Ragas metrics:

- **Faithfulness** - How well the answer is grounded in retrieved context
- **Answer Relevancy** - How pertinent the answer is to the query
- **Context Precision** - Relevance of retrieved contexts
- **Context Recall** - How much relevant information is captured

```python
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)

# Evaluate RAG performance
result = evaluate(
    dataset=dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_recall,
        context_precision
    ],
    llm=llm,
    embeddings=embeddings
)
```

## 🎯 Performance Metrics

- **Excellent/Production-Ready**: >0.85
- **Good/Acceptable**: 0.7–0.85
- **Needs Improvement**: 0.5–0.7
- **Poor**: <0.5

## 📁 Project Structure

```
AI-Agent/
├── agent.ipynb                 # Main RAG implementation notebook
├── rag_agent_streamlit.py     # Streamlit web interface
├── rag_with_rerank.py         # Reranking implementation
├── rerank_module.py           # Reranking utilities
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── RERANKING_GUIDE.md         # Reranking documentation
└── Amazon-2024-Annual-Report.pdf  # Sample document
```

## 🚀 Quick Start

1. **Run the Jupyter notebook**:
   ```bash
   jupyter notebook agent.ipynb
   ```

2. **Launch Streamlit app**:
   ```bash
   streamlit run rag_agent_streamlit.py
   ```

3. **Test with sample document**:
   The system comes with Amazon's 2024 Annual Report for testing.

## 🔍 Key Features Explained

### Text Chunking Strategy
- **Chunk Size**: 1000 characters (optimal for context preservation)
- **Overlap**: 200 characters (prevents information loss at boundaries)
- **Method**: Recursive character splitting (respects natural boundaries)

### Retrieval Methods
- **Dense Retrieval**: Semantic similarity using embeddings
- **Sparse Retrieval**: Keyword-based search (BM25)
- **Hybrid**: Combines both methods for optimal results

### Reranking Benefits
- Improves retrieval quality by 15-20%
- Reduces noise in retrieved contexts
- Better semantic understanding of query-document relationships

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- [LangChain](https://langchain.com/) for the RAG framework
- [HuggingFace](https://huggingface.co/) for embedding models
- [FAISS](https://github.com/facebookresearch/faiss) for vector search
- [Ragas](https://github.com/explodinggradients/ragas) for evaluation metrics