# Multi-LLM RAG-Enhanced ChatBot with In-Built Search

## Setup

```bash
pip install -r requirements.txt
```

## Add PDFs

Put your PDF files inside the `pdfs/` directory.

## Run App

```bash
streamlit run app.py
```

Make sure Ollama is running and all 3 models are pulled:
```bash
ollama run phi
ollama run llama2
ollama run mistral
```
# Description:
A highly customizable conversational AI ChatBot integrated with Retrieval-Augmented Generation (RAG), designed to deliver accurate, context-aware answers using multiple LLMs (Language Models). This system incorporates:
In-built search tools (e.g., DuckDuckGo, Wikipedia, or web scraping) for real-time external information retrieval.
Vector database (e.g., ChromaDB) for semantic document search and memory.
Multi-LLM architecture allowing dynamic selection between lightweight local models (like Phi, TinyLlama, or Gemma) for efficient and diverse response generation.

# Key Features:
RAG Pipeline: Enhances LLM responses using retrieved context from internal docs and external web tools.
Custom Tool Integration: Includes DuckDuckGo, Wikipedia search, and scraping tools to fetch real-time data.
LLM Switching: Routes queries through different LLMs for experimentation, fallback, or ensemble-based reasoning.
PDF/Text Loader: Supports custom knowledge ingestion via uploaded .pdf or .txt files.
Fast, Local, Offline-Friendly: Uses Ollama-based LLMs to reduce cost and increase privacy.

# Tools Used
Language Models: Ollama LLMs, CrewAI Agents
Vector Database: ChromaDB
Embeddings: HuggingFace (all-MiniLM-L6-v2)
Search & Retrieval: Wikipedia, DuckDuckGo
Monitoring: LangSmith (prompt & token tracking)
Demo Interface: Streamlit / Web App
Libraries: LangChain
