# SecurityRAG

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![RAG](https://img.shields.io/badge/AI-RAG-green)
![Cybersecurity](https://img.shields.io/badge/Domain-Cybersecurity-orange)

AI-powered Security Knowledge Assistant using Retrieval-Augmented Generation (RAG).

## Overview

SecurityRAG enables SOC analysts, DFIR analysts, and security engineers to ask natural language questions and receive contextual answers — with source citations — from trusted cybersecurity knowledge sources.

## Architecture

```
User Query
    ↓
Retriever
    ↓
Vector Database
    ↓
Security Documents
    ↓
LLM
    ↓
Answer + Source Citations
```

## Features (Planned)

- Natural language security queries
- Vector-based semantic search
- MITRE ATT&CK mapping
- Source citations
- Threat investigation assistance
- Incident response guidance
- Security documentation retrieval

## Phase 2 Features

- PDF ingestion
- Document chunking

## Phase 3 Features

- Semantic embeddings
- Query embeddings
- Sentence Transformer support

## Phase 4 Features

- ChromaDB vector storage
- Persistent collections
- Similarity search

## Phase 5 Features

- Semantic retrieval
- Context generation
- Similarity search

## Tech Stack

- Python 3.11
- Streamlit
- LangChain
- ChromaDB
- OpenAI API
- Sentence Transformers

## Project Structure

```
SecurityRAG/
├── app/
│   ├── app.py                       # Streamlit application entry point
│   ├── config.py                    # Centralized configuration
│   ├── ingestion/
│   │   └── document_loader.py       # Document ingestion interface
│   ├── embeddings/
│   │   └── embedding_model.py       # Embedding model interface
│   ├── vectorstore/
│   │   └── chroma_store.py          # Vector store interface
│   ├── retrieval/
│   │   └── retriever.py             # Retrieval pipeline interface
│   └── prompts/
│       └── security_prompt.txt      # LLM prompt template
├── data/                            # Raw/processed security knowledge sources
├── docs/                            # Project documentation
├── tests/                           # Unit tests
├── requirements.txt
├── .gitignore
├── .env.example
└── README.md
```

## Installation

### Prerequisites

- Python 3.11
- An OpenAI API key

### Steps

1. Clone the repository:

   ```bash
git clone https://github.com/rishikumarbommakanti-ops/SecurityRAG.git
cd SecurityRAG
```

2. Create and activate a virtual environment:

   ```bash
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

   ```bash
pip install -r requirements.txt
```

4. Configure environment variables:

   ```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

5. Run the Streamlit application:

   ```bash
streamlit run app/app.py
```

## Status

This repository is currently in **Phase 1**: project skeleton and boilerplate only.
Document ingestion, embedding generation, vector storage, and retrieval logic are defined as placeholder interfaces and will be implemented in later phases.

## License

TBD
