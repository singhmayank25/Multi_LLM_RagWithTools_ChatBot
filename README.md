# Multi-LLM RAG-Enhanced ChatBot with In-Built Search

## 📦 Setup

```bash
pip install -r requirements.txt
```

## 📄 Add PDFs

Put your PDF files inside the `pdfs/` directory.

## 🚀 Run App

```bash
streamlit run app.py
```

Make sure Ollama is running and all 3 models are pulled:
```bash
ollama run phi
ollama run llama2
ollama run mistral
```

## 💡 Features

- Compare RAG responses from 3 different models side by side.
- Uses InstructorEmbeddings + ChromaDB + LangChain.
