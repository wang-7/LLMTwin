# LLMTwin

LLMTwin is a comprehensive framework for building and fine-tuning Large Language Models (LLMs) with Retrieval Augmented Generation (RAG) capabilities. The project provides a complete pipeline for document processing, embedding, storage, and LLM fine-tuning.

## Overview

LLMTwin is designed to help developers and researchers build custom LLM applications by providing:

- **Document Processing Pipeline**: Clean, chunk, and embed documents for RAG systems
- **Multi-Modal Storage**: Integration with MongoDB for document metadata and Qdrant for vector embeddings
- **LLM Fine-tuning**: Tools for fine-tuning LLMs using PEFT (Parameter Efficient Fine-Tuning) techniques
- **RAG Integration**: Ready-to-use RAG components with embedding and reranking capabilities

## Features

- **Document Processing**: Clean, chunk, and process various document types
- **Vector Storage**: Efficient vector storage and retrieval using Qdrant
- **Metadata Management**: Document metadata stored in MongoDB
- **LLM Fine-tuning**: Easy fine-tuning of LLMs with LoRA
- **RAG Pipeline**: Complete RAG pipeline with embedding and reranking
- **Modular Architecture**: Clean, modular codebase following domain-driven design
- **Configuration Management**: Flexible configuration using Pydantic settings

## Architecture

The project is organized into several key components:

### Core Components
- **Domain**: Domain models and business logic
- **Preprocess**: Document cleaning, chunking, and preprocessing pipeline
- **Feature Engineering**: Embedding and feature extraction
- **Networks**: LLM integration and fine-tuning components
- **Connections**: Database connection management (MongoDB, Qdrant)

### Key Technologies
- **LLM Framework**: Unsloth for efficient LLM fine-tuning
- **Vector Database**: Qdrant for vector storage and similarity search
- **Document Database**: MongoDB for metadata storage
- **Embedding Models**: Sentence Transformers for text embeddings
- **Fine-tuning**: PEFT (LoRA) for efficient model adaptation

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd LLMTwin

# Install dependencies using uv (recommended)
uv sync

# Or install using pip
pip install -e .
```

## Usage

### 1. Document Processing Pipeline

The framework provides a complete pipeline for processing documents:

```python
from llmtwin.preprocess import clean_handler, chunk_handler, embed_handler

# Clean documents
cleaned_docs = clean_handler.clean_documents(raw_documents)

# Chunk documents
chunks = chunk_handler.chunk_documents(cleaned_docs)

# Embed documents
embedded_chunks = embed_handler.embed_documents(chunks)
```

### 2. LLM Fine-tuning

Fine-tune LLMs using the built-in fine-tuning pipeline:

```python
from llmtwin.finetune import finetune

# Define configuration
class LoadModelConfig:
    max_seq_length: int = 2048
    dtype = None
    load_in_4bit = False

class PeftModelConfig:
    r: int = 16
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    lora_alpha: int = 16
    lora_dropout: float = 0
    bias: str = "none"

# Fine-tune the model
loadconfig = LoadModelConfig()
peftconfig = PeftModelConfig()
finetune(loadconfig, peftconfig)
```

### 3. Database Connections

The framework manages connections to both MongoDB and Qdrant:

```python
from llmtwin.connection import MongoDBConnection, QdrantConnection

# Get MongoDB connection
mongo_client = MongoDBConnection()

# Get Qdrant connection
qdrant_client = QdrantConnection()
```

## Configuration

The project uses Pydantic Settings for configuration management. You can customize the following settings:

- `DATABASE_NAME`: MongoDB database name (default: 'LLMTwinDB')
- `TEXT_EMBEDDING_MODEL_ID`: Embedding model ID (default: 'sentence-transformers/all-MiniLM-L6-v2')
- `RERANKING_CROSS_ENCODER_MODEL_ID`: Reranking model ID (default: 'cross-encoder/ms-marco-MiniLM-L-4-v2')
- `RAG_MODEL_DEVICE`: Device for RAG models (default: 'cpu')

## Project Structure

```
├── notebooks/               # Jupyter notebooks for experimentation
│   ├── hf_datasets.ipynb    # Hugging Face datasets examples
│   ├── hf_transformers.ipynb # Hugging Face transformers examples
│   ├── mongodb_test.ipynb   # MongoDB integration tests
│   ├── qdrant_test.ipynb    # Qdrant vector database tests
│   └── unsloth.ipynb        # Unsloth LLM fine-tuning examples
├── script/                  # Utility scripts
├── src/
│   └── llmtwin/             # Main package
│       ├── domain/          # Domain models and business logic
│       │   ├── base/        # Base classes
│       │   ├── document.py  # Document models
│       │   ├── chunk_document.py # Chunked document models
│       │   ├── cleaned_document.py # Cleaned document models
│       │   ├── data_category.py # Data categorization
│       │   └── embed_document.py # Embedded document models
│       ├── preprocess/      # Document preprocessing pipeline
│       │   ├── clean_handler.py # Document cleaning
│       │   ├── chunk_handler.py # Document chunking
│       │   ├── embed_handler.py # Document embedding
│       │   └── dispatcher.py    # Pipeline dispatcher
│       ├── feature_engineering/ # Feature extraction and engineering
│       ├── networks/        # LLM and neural network components
│       ├── connection.py    # Database connection management
│       ├── finetune.py      # LLM fine-tuning utilities
│       └── settings.py      # Configuration management
├── tests/                   # Test suite
├── pyproject.toml           # Project dependencies and configuration
└── README.md                # Project documentation
```

## Dependencies

The project uses the following key dependencies:

- `unsloth`: Efficient LLM fine-tuning
- `pymongo`: MongoDB integration
- `qdrant-client`: Vector database client
- `sentence-transformers`: Text embedding models
- `pydantic`: Data validation and settings management
- `loguru`: Logging
- `modelscope`: Model hub integration

## Development

To contribute to the project:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Run tests: `pytest`
6. Submit a pull request

### Running Tests

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=llmtwin
```

### Code Quality

The project uses Ruff for linting and formatting:

```bash
# Lint code
ruff check .

# Format code
ruff format .
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

We welcome contributions! Please see our contributing guidelines for more information.

## Acknowledgments

- Thanks to the Unsloth team for efficient LLM fine-tuning
- Thanks to the Qdrant team for the vector database
- Thanks to the Sentence Transformers team for embedding models