import streamlit as st
import os
import glob
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()



st.set_page_config(page_title="Mbah AI Primbon", page_icon="🔮", layout="centered")

# CSS
st.markdown("""
<style>
  .stApp {
        background-image: url("https://umj.ac.id/storage/2022/02/4371933-scaled.jpg");
        background-size: cover;
        background-attachment: fixed;
        background-repeat: repeat;
        opacity: 1.0;
        z-index: -1;
 }
 
 /* Biar teks nggak ketutup */
 .main .block-container {
        background-color: rgba(255, 248, 231, 0.93);
        padding: 2rem;
        border-radius: 15px;
        border: 3px solid #8B4513;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
 }
 
 .main-header {
        font-size: 3rem !important;
        color: red;
        margin: 0;
        font-weight: 800 !important;
        line-height: 1.1;
        text-shadow: 2px 2px 4px rgba(255,255,255,0.8);
    }
 .sub-header {
        color: white;
        margin: 0;
        font-size: 1.1rem;
        text-align: left;
        font-weight: 600;
    }
            
.jawaban-box {
        background-color: #FFF8E1;
        border: 2px dashed #A1887F;
        border-radius: 10px;
        padding: 20px;
        margin-top: 15px;
        color: black;
        font-size: 1.1rem;
        line-height: 1.6;
        
 }
 
 /* Kotak input */
 .stTextInput > div > div > input {
        background-color: #FFF8E1;
        border: 2px solid #8B4513;
        border-radius: 8px;
 }
 .caption {
        color: red;
        font-size: 2rem;
        text-shadow: black;
        font-weight: 600;
        text-align: center;
        
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
    # Baca semua file.txt di folder
    all_texts = []
    for file_path in glob.glob("*.txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            isi = f.read().lower()
            all_texts.append(isi)
            

    # Gabungin semua jadi satu
    full_text = "\n\n---BATAS FILE---\n\n".join(all_texts)
    
    # FAISS
    splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
    docs = splitter.create_documents([full_text])
    
    db = FAISS.from_documents(docs, HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-small"))

    # MODEL BARU YANG AKTIF. Pilih salah satu:
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3) 
    # llm = ChatGroq(model="llama3-70b-8192", temperature=0.3) # Cadangan
    # llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0.3) # Paling cepet

    return db, llm

db, llm = load_bot()

q = st.text_input("Tulis wetonmu di sini!")

if q:
    with st.spinner('Mbah lagi buka kitab dulu...'):
        try:
            context = db.similarity_search(q.lower(), k=6)
            
            prompt = f"""Kamu adalah Mbah Primbon. Tugasmu hanya membaca data primbon di bawah dan menjawab pertanyaan user.
            
ATURAN KERAS :
1. JANGAN mengarang atau pakai pengetahuan luarmu.
2. JANGAN menjumlahkan neptu sendiri. Langsung pakai data yang ada.
3. Jika di data tidak ada jawaban, cari sumber lain yang falid
4. Jawab dengan gaya yang bisa merangkul + humoris + berikan nasehat.

DATA PRIMBON:
{context}

PERTANYAAN USER: {q}

JAWABAN MBAH:"""
            
            jawaban = llm.invoke(prompt).content # <-- variabel jawaban dibuat di sini
            
            # Masukin jawaban ke kotak, aman karena 'jawaban' udah pasti ada
            st.markdown(f'<div class="jawaban-box">{jawaban}</div>', unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Aduh, kitab e Mbah sobek Le: {e}")
        
        # st.caption("⚠️ Disclaimer : Hanya untuk pengetahuan tentang kearifan lokal, Karena semua yang terjadi adalah ketentuan Allah")
        st.caption('<p class="caption">⚠️ Disclaimer : Hanya untuk pengetahuan tentang kearifan lokal, Karena semua yang terjadi adalah ketentuan Allah</p>', unsafe_allow_html=True)