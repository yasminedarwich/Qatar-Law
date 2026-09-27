import streamlit as st
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# ---------- Meezan law page URLs (add more as needed) ----------
LAW_URLS = {
    "Labor Law": "https://www.meezan.qa/law/14/2004",
    "Penal Code": "https://www.meezan.qa/law/11/2004",
    "Commercial Law": "https://www.meezan.qa/law/27/2006",
    "Civil Code": "https://www.meezan.qa/law/22/2004",
    "Family Law": "https://www.meezan.qa/law/29/2006",
}

# ---------- Scrape a law page (cached) ----------
@st.cache_data(show_spinner=False)
def scrape_law(url: str) -> str:
    """Fetch and extract text from a Meezan law page."""
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        # Try common Meezan content containers
        content = soup.find("div", class_="law-content") or soup.find("article") or soup.find("main")
        if content:
            return content.get_text(separator="
", strip=True)
        else:
            return "Could not extract content."
    except Exception as e:
        return f"Error: {e}"

# ---------- Build vector store from scraped laws ----------
@st.cache_resource(show_spinner=True)
def build_vectorstore():
    """Scrape all law pages and return a FAISS vector store."""
    all_texts = []
    for law_name, url in LAW_URLS.items():
        text = scrape_law(url)
        if text and "Error" not in text and "Could not extract" not in text:
            all_texts.append(f"--- {law_name} ---
{text}")
    if not all_texts:
        return None

    full_text = "

".join(all_texts)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([full_text])

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Qatar Law Genius", layout="wide")
st.title("🇶🇦 Qatar Law Genius – 100% Free & Accurate")
st.markdown("Answers from official Meezan texts. No downloads, no paid APIs.")

# Build vector store
with st.spinner("Fetching and indexing Meezan laws..."):
    vectorstore = build_vectorstore()
if vectorstore is None:
    st.error("Could not fetch any law pages. Check URLs or internet connection.")
    st.stop()

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Local LLM (Ollama)
llm = Ollama(model="llama3", temperature=0)

# Prompt forcing answer from context only
prompt_template = """You are an expert on Qatar law. Answer ONLY using the provided legal texts.
If the answer is not in the texts, say "I don't have information on that from Meezan."

Context:
{context}

Question: {question}
Answer:"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)

# User input
query = st.text_input("Ask a legal question (e.g., 'What is the penalty for theft?')")

if query:
    with st.spinner("Searching Meezan..."):
        result = qa_chain({"query": query})
        answer = result["result"]
        sources = result["source_documents"]

    st.subheader("Answer")
    st.write(answer)

    with st.expander("📄 View source texts"):
        for i, doc in enumerate(sources, 1):
            st.markdown(f"**Source {i}:** `{doc.metadata.get('source', 'Unknown')}`")
            st.text(doc.page_content[:300] + "...")
