import streamlit as st
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

if "GROQ_API_KEY" in st.secrets:
    
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
else:
    
    from dotenv import load_dotenv
    load_dotenv()

st.set_page_config(page_title="Mbah AI Primbon", page_icon="🔮", layout="centered")

# CSS
st.markdown("""
<style>
 .stApp {
        background-image: url("https://i.imgur.com/link-gambar-batikmu.jpg");
        background-size: cover;
        background-attachment: fixed;
 .main-header {
        font-size: 3.5rem !important;
        color: black;
        margin: 0;
        font-weight: 800rem !important;
        line-height: 1.1;
    }
 .sub-header {
        color: #8B7355;
        margin: 0;
        font-size: 1.1rem;
        position: center;
    }
</style>
""", unsafe_allow_html=True)

# HEADER
_, col_main, _ = st.columns([0.5, 3, 0.5])
with col_main:
    col_gambar, col_tulisan = st.columns([1, 4])
    with col_gambar:
        st.image("mbah.png", width=60)
    with col_tulisan:
        st.markdown('<p class="main-header">Ngaji Fathul Jare</p>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">🔮 Bersama Mbah AI - Pituduh Jawa Digital</p>', unsafe_allow_html=True)

st.divider()


@st.cache_resource
def load_bot():
    with open('primbon.txt', 'r', encoding='utf-8') as f: text = f.read()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.create_documents([text])
    db = Chroma.from_documents(docs, HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))

    # MODEL BARU YANG AKTIF. Pilih salah satu:
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3) 
    # llm = ChatGroq(model="llama3-70b-8192", temperature=0.3) # Cadangan
    # llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0.3) # Paling cepet

    return db, llm

db, llm = load_bot()
q = st.text_input("Tulis wetonmu, misal: Minggu Kliwon. Atau tanya: Jodohku Sabtu Legi cocok nggak?")

if q:
    with st.spinner('Mbah lagi buka kitab Betaljemur...'):
        context = db.similarity_search(q, k=2)
        prompt = f"Kamu sesepuh Jawa yang bijak, halus, & ada humor dikit. Jawab pakai bahasa Indonesia. Jangan nakut-nakutin. Berdasarkan data primbon ini: {context}. Pertanyaan user: {q}. Kasih saran praktis di akhir."
        st.write(llm.invoke(prompt).content)
        st.caption("⚠️ Hanya untuk hiburan, Karena semua yang terjadi adalah ketentuan Allah")