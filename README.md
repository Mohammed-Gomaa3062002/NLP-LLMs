# RAG Pipeline for Document Retrieval

This project implements a **Retrieval-Augmented Generation (RAG)** system for general document retrieval using fine-tuned Large Language Models (LLMs). The system integrates **ChromaDB** for vector storage and **Ollama embeddings** (BAAI/bge-small-en, TinyLlama) for semantic search. It features a modular pipeline for data ingestion, chunking, embedding, retrieval, and generation, and deploys **Llama2** locally via Ollama for cost-efficient inference.

## Features

- **Retrieval-Augmented Generation (RAG)**: Combines document retrieval with fine-tuned LLMs for accurate and context-aware responses.
- **ChromaDB Integration**: Efficient vector storage for fast and scalable document retrieval.
- **Ollama Embeddings**: Utilizes BAAI/bge-small-en and TinyLlama embeddings for semantic search.
- **Modular Pipeline**:  
  - **Data Ingestion**: Load and preprocess documents.  
  - **Chunking**: Split documents into manageable chunks.  
  - **Embedding**: Generate embeddings for semantic search.  
  - **Retrieval**: Retrieve relevant documents using ChromaDB.  
  - **Generation**: Generate responses using fine-tuned LLMs (Llama2).
- **Local Deployment**: Deploys Llama2 locally via Ollama for cost-efficient and low-latency inference.

## Technologies Used

- **Python**: Primary programming language.
- **LlamaIndex**: Framework for building RAG pipelines.
- **ChromaDB**: Vector database for document storage and retrieval.
- **Ollama**: Local deployment of LLMs (Llama2) and embeddings.
- **HuggingFace**: Pre-trained models and embeddings (BAAI/bge-small-en).
- **LangChain**: Integration for document retrieval and generation.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/RAG-Pipeline-Document-Retrieval.git
   cd RAG-Pipeline-Document-Retrieval
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up Ollama for local LLM deployment:
   - Download and install Ollama from [here](https://ollama.ai/).
   - Pull the Llama2 model:
     ```bash
     ollama pull llama2
     ```

4. Configure environment variables:
   - Create a `.env` file and add your OpenAI API key (if needed):
     ```bash
     OPENAI_API_KEY=your-api-key
     ```

## Usage

1. **Data Ingestion**:
   - Place your documents in the `data/` directory.
   - Run the ingestion script:
     ```bash
     python ingestion.py
     ```

2. **Query the RAG System**:
   - Run the query engine:
     ```bash
     python query_engine.py
     ```
   - Enter your query when prompted.

## Example Queries

- "What is the history of NLP?"
- "Explain the concept of retrieval-augmented generation."
- "How does ChromaDB work for vector storage?"

## Project Structure

```
RAG-Pipeline-Document-Retrieval/
├── data/                  # Directory for input documents
├── ingestion.py           # Script for data ingestion and preprocessing
├── query_engine.py        # Script for querying the RAG system
├── requirements.txt       # List of dependencies
├── README.md              # Project documentation
└── .env                   # Environment variables (e.g., API keys)
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

**Author**: Mohamed Gomaa  
**Contact**: mogommaa2002@gmail.com  

--- 
