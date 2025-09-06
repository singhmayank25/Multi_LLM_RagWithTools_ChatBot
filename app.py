import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama
from ddgs import DDGS
from langchain.chains import ConversationalRetrievalChain
from langchain.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv

load_dotenv()

os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_PROJECT"]="RAG_ChatBot"

with DDGS() as ddgs:
    results = [r for r in ddgs.text("LangChain tutorial")]
    print(results[:3])
doc=PyPDFLoader("attention.pdf")
documents = doc.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(documents)
print(f"Split into {len(docs)} chunks.")

embedding_model = HuggingFaceInstructEmbeddings(
    model_name="hkunlp/instructor-large",
    model_kwargs={"device": "cpu"} 
)
persist_directory = "./chroma_db"

if os.path.exists(persist_directory) and os.path.isdir(persist_directory):
    print("Loading existing Chroma vector store...")
    db = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model
    )
else:
    db = Chroma.from_documents(
        documents=docs,
        embedding=embedding_model,
        persist_directory="./chroma_db"
    )
    db.persist()
    print("ChromaDB created and saved.")

retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

hindi_prompt = PromptTemplate(
    input_variables=["question"],
    template="""
नीचे दिए गए संदर्भ का उपयोग करके उपयोगकर्ता के प्रश्न का उत्तर हिंदी में दें।

प्रश्न:
{question}

उत्तर हिंदी में दें:
"""
)

prompt = PromptTemplate(
    input_variables=["question"],
    template="""
You are a helpful AI assistant. Please answer the following question:

Question: {question}
Answer:
"""
)
phi_chain = ConversationalRetrievalChain.from_llm(llm=ChatOllama(model="phi"), retriever=retriever)
tinyllama_chain = LLMChain(llm=ChatOllama(model="tinyllama"), prompt= hindi_prompt)
gemma_chain = LLMChain(llm=ChatOllama(model="gemma:2b"),prompt= prompt)


#Tools
duckduckgo = DuckDuckGoSearchRun()
wikipedia_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())


tools = [
    Tool(name="duckduckgo", func=duckduckgo.run, description="Search the web"),
    Tool(name="wikipedia", func=wikipedia_tool.run, description="Search Wikipedia"),
]


tool_agent = initialize_agent(tools, ChatOllama(model="tinyllama"), agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,    max_execution_time=15,  # ⏱️ seconds
    max_iterations=25, handle_parsing_errors=True,verbose=False)


# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="RAG with 3 LLMs", layout="wide")
st.title("🧠Multi-RAG Chatbot along with 🔍 Wikipedia and DuckDuckGo")

query = st.text_input("Ask a question:", placeholder="e.g., What is LangChain?")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if query:
    with st.spinner("Running all models..."):
        phi_ans = phi_chain.invoke({"question": query, "chat_history": st.session_state.chat_history})["answer"]
        tinyllama_ans = tinyllama_chain.invoke({"question": query})["text"]
        gemma_ans = gemma_chain.invoke({"question": query})["text"]
        try:
            tool_ans = tool_agent.run(query)
        except Exception as e:
            tool_ans = f"Tool Error: {e}"

    st.session_state.chat_history.append((query, phi_ans))  # Optionally track only one

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("🤖 PHI")
        st.write(phi_ans)
    with col2:
        st.subheader("🦙 TINYLLAMA")
        st.write(tinyllama_ans)
    with col3:
        st.subheader("⚡ GEMMA:2b")
        st.write(gemma_ans)
    st.markdown("---")
    st.subheader("🔎 Tool Agent Output (Wikipedia, DuckDuckGo)")
    st.info(tool_ans)
