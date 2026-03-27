import streamlit as st
import os
import tempfile
import time
from dotenv import load_dotenv

import chromadb
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

# Streamlit Page Config
st.set_page_config(
    page_title="RAG Analyzer - Precision Architect", 
    page_icon="https://cdn-icons-png.flaticon.com/512/2632/2632280.png", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- NEW DARK THEME TAILWIND & CSS ---
st.markdown("""
<script src="https://cdn.tailwindcss.com?plugins=forms,container-queries"></script>
<link href="https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;700&family=Inter:wght@400;500;700&family=Space+Grotesk:wght@400;500;700&display=swap" rel="stylesheet"/>
<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&display=swap" rel="stylesheet"/>
<script>
tailwind.config = {
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        "primary-fixed-dim": "#7f8af6",
        "on-secondary": "#33005b",
        "on-secondary-fixed-variant": "#7511c3",
        "secondary-dim": "#9c48ea",
        "on-background": "#dee5ff",
        "tertiary-fixed": "#ff8ed2",
        "surface-container-high": "#141f38",
        "secondary-container": "#6f00be",
        "surface-variant": "#192540",
        "surface-container-highest": "#192540",
        "error-dim": "#d73357",
        "surface-container-low": "#091328",
        "on-error": "#490013",
        "tertiary-container": "#ff8ed2",
        error: "#ff6e84",
        "secondary-fixed-dim": "#dbb4ff",
        "on-surface": "#dee5ff",
        "on-tertiary-container": "#63054a",
        "on-tertiary": "#701455",
        "primary-fixed": "#8d98ff",
        "inverse-primary": "#4954bc",
        "surface-tint": "#9fa7ff",
        "outline-variant": "#40485d",
        primary: "#9fa7ff",
        "inverse-on-surface": "#4d556b",
        "on-primary-fixed": "#000000",
        "tertiary-fixed-dim": "#ef81c4",
        "surface-container": "#0f1930",
        "on-surface-variant": "#a3aac4",
        "on-tertiary-fixed-variant": "#6e1354",
        surface: "#060e20",
        "on-primary": "#101b8b",
        background: "#060e20",
        "on-error-container": "#ffb2b9",
        "on-secondary-container": "#e9cdff",
        "surface-bright": "#1f2b49",
        "on-secondary-fixed": "#4f0089",
        "surface-dim": "#060e20",
        secondary: "#c180ff",
        tertiary: "#ffa5d9",
        "on-tertiary-fixed": "#3b002b",
        "error-container": "#a70138",
        "on-primary-container": "#000a7b",
        "primary-dim": "#8a95ff",
        outline: "#6d758c",
        "surface-container-lowest": "#000000",
        "secondary-fixed": "#e5c6ff",
        "tertiary-dim": "#ef81c4",
        "inverse-surface": "#faf8ff",
        "on-primary-fixed-variant": "#0c1889",
        "primary-container": "#8d98ff"
      },
      fontFamily: {
        headline: ["Manrope"],
        body: ["Inter"],
        label: ["Space Grotesk"]
      }
    }
  }
};
</script>
<style>
    .stApp {
        background-color: #060e20;
        color: #dee5ff;
    }
    
    .block-container {
        padding: 0 !important;
        max-width: 100% !important;
    }
    
    /* [data-testid="stHeader"] { display: none !important; } */
    #MainMenu { visibility: hidden; }

    /* Docked Sidebar */
    [data-testid="stSidebar"] {
        background-color: #091328 !important;
        border-right: 1px solid rgba(255,255,255,0.05);
        width: 280px !important;
    }
    [data-testid="stSidebarNav"] { display: none !important; }

    /* Main Content */
    .main-content {
        padding: 3rem 4rem;
        max-width: 1400px;
    }

    /* Fixed Header Styling */
    .custom-header {
        background-color: #060e20;
        width: 100%;
        padding: 1rem 2rem;
        display: flex !important;
        justify-content: space-between !important;
        align-items: center !important;
        border-bottom: 1px solid rgba(255,255,255,0.05);
    }
    
    .profile-img {
        width: 32px !important;
        height: 32px !important;
        border-radius: 9999px !important;
        border: 1px solid rgba(159, 167, 255, 0.2);
    }

    /* Sidebar Buttons */
    section[data-testid="stSidebar"] .stButton>button {
        background-color: transparent !important;
        color: #a3aac4 !important;
        border: none !important;
        text-align: left !important;
        padding: 0.8rem 1.2rem !important;
        width: 100% !important;
        display: flex !important;
        align-items: center !important;
        gap: 12px !important;
        font-family: 'Manrope', sans-serif !important;
    }
    section[data-testid="stSidebar"] .stButton>button:hover {
        background-color: rgba(255,255,255,0.05) !important;
        color: #9fa7ff !important;
    }

    /* Global Scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #060e20; }
    ::-webkit-scrollbar-thumb { background: #192540; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# --- BACKEND ---
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@st.cache_resource
def get_llm():
    return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)

embeddings = get_embeddings()
DB_DIR = "./chroma_db"

if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Retriever"
if "process_status" not in st.session_state:
    st.session_state.process_status = "Ready"

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("""
    <div class="mb-10 pt-4">
        <div class="flex items-center gap-3 mb-10">
            <div class="w-10 h-10 bg-gradient-to-br from-[#9fa7ff] to-[#c180ff] rounded-lg flex items-center justify-center text-[#101b8b]">
                <span class="material-symbols-outlined" style="font-variation-settings: 'FILL' 1;">psychology</span>
            </div>
            <div>
                <h2 class="text-indigo-400 font-bold text-sm">Precision Architect</h2>
                <p class="text-[9px] text-[#a3aac4] uppercase tracking-tighter">RAG Engine v2.4</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("Dashboard", key="nav_dash"): st.session_state.active_tab = "Dashboard"
    if st.button("Retriever", key="nav_retr"): st.session_state.active_tab = "Retriever"
    if st.button("Verify DB", key="nav_verify"): st.session_state.active_tab = "Verify DB"
    
    st.markdown("""
    <div class="mt-80 pt-6 border-t border-white/5">
        <div class="flex items-center gap-3 text-[#a3aac4] text-xs">
            <span class="material-symbols-outlined text-sm">bolt</span>
            <span>System Health: Optimal</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- CUSTOM HEADER ---
st.markdown(f"""
<div class="custom-header font-manrope">
    <div class="flex items-center gap-8">
        <span class="text-2xl font-bold tracking-tighter text-indigo-400">RAG Analyzer</span>
        <div class="hidden md:flex gap-6">
            <span class="text-[#a3aac4] text-sm">Portfolio</span>
            <span class="text-indigo-400 text-sm border-b-2 border-indigo-400 pb-1">Analyses</span>
            <span class="text-[#a3aac4] text-sm">Sources</span>
        </div>
    </div>
    <div class="flex items-center gap-4">
        <div class="flex items-center gap-2 bg-[#0f1930] px-3 py-1.5 rounded-full border border-[#40485d]/20">
            <span class="relative flex h-2 w-2">
                <span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-[#9fa7ff] opacity-75"></span>
                <span class="relative inline-flex rounded-full h-2 w-2 bg-[#9fa7ff]"></span>
            </span>
            <span class="text-[10px] uppercase tracking-widest text-[#9fa7ff] font-bold font-label">Status: {st.session_state.process_status}</span>
        </div>
        <img src="https://lh3.googleusercontent.com/aida-public/AB6AXuCHg42IWLey4S15SqkykRxyXZ74kddwgq21vEX9uiG7QWnY_jlHlpQHBelnSrAMrvANaeXGQBJZ2qZn0f6kOOvLsoq7Q-cq379cmg6TdqL4rT_0XzNRWi4dIwrFilyDGKhr5yCyohoXL1xviVF5bUuRt8Yifxe-VxcnY4D8YLCtxxYVNw8POr7pS-0JH8PXge7dq7HkrecSoLkD56oNshJ_iv4w7kMyv3riQuGn5LZXVdB1-NXyu6BOa6vvncteixXZfnbw3ewfQKE" class="profile-img">
    </div>
</div>
""", unsafe_allow_html=True)

# --- MAIN CONTENT ---
st.markdown('<div class="main-content">', unsafe_allow_html=True)

if st.session_state.active_tab == "Retriever":
    st.markdown("""
    <div class="space-y-12">
        <div class="space-y-2">
            <h1 class="text-4xl font-extrabold font-headline tracking-tighter text-[#dee5ff]">Knowledge Retrieval</h1>
            <p class="text-[#a3aac4] text-lg">Initialize your analysis by processing financial literature and investment reports.</p>
        </div>
    """, unsafe_allow_html=True)
    
    r_cols = st.columns([2, 1], gap="large")
    with r_cols[0]:
        uploaded_file = st.file_uploader("Upload PDF", type="pdf", label_visibility="collapsed")
        st.markdown("""
        <div class="bg-[#091328] p-16 rounded-xl border-2 border-dashed border-[#40485d]/30 flex flex-col items-center justify-center text-center space-y-4 group hover:border-[#9fa7ff]/50 transition-all cursor-pointer">
            <div class="w-16 h-16 bg-[#192540] rounded-full flex items-center justify-center text-[#9fa7ff] group-hover:scale-110 transition-transform">
                <span class="material-symbols-outlined text-3xl">upload_file</span>
            </div>
            <div>
                <h3 class="text-xl font-bold font-headline text-[#dee5ff]">Drag investment docs here</h3>
                <p class="text-[#a3aac4]">Supports PDF, DOCX up to 50MB</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with r_cols[1]:
        st.markdown(f"""
        <div class="bg-[#0f1930] p-8 rounded-xl flex flex-col justify-between h-full">
            <div class="space-y-4">
                <div class="flex items-start justify-between">
                    <div class="flex items-center gap-3">
                        <div class="w-10 h-10 bg-[#a70138]/20 text-[#ff6e84] rounded flex items-center justify-center">
                            <span class="material-symbols-outlined">picture_as_pdf</span>
                        </div>
                        <div>
                            <p class="font-bold text-[#dee5ff] text-sm truncate w-32">{"File selected" if uploaded_file else "No file selected"}</p>
                            <p class="text-[10px] font-label text-slate-500">{(uploaded_file.size/1024/1024) if uploaded_file else 0:.1f} MB</p>
                        </div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        if uploaded_file and st.button("Process Document", key="btn_p"):
            with st.spinner("Analyzing..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.getvalue()); tmp_path = tmp.name
                try:
                    loader = PyPDFLoader(tmp_path); docs = loader.load()
                    splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
                    Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=DB_DIR)
                    st.session_state.process_status = "READY"; st.success("Indexing Complete")
                finally: os.unlink(tmp_path)
        
        st.markdown("""
        <button class="w-full bg-[#1f2b49] text-[#9fa7ff] font-bold py-3 rounded-lg hover:bg-[#9fa7ff] hover:text-[#101b8b] transition-all active:scale-95 mt-4">
            Process Document (Click above button)
        </button>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

elif st.session_state.active_tab == "Verify DB":
    st.markdown("""
    <div class="flex items-center justify-between mb-8">
        <h2 class="text-2xl font-extrabold font-headline tracking-tight">Backend Verification</h2>
        <div class="flex gap-2">
            <span class="px-3 py-1 bg-[#192540] rounded-full text-[10px] font-label uppercase text-slate-400">Latent Space v2</span>
            <span class="px-3 py-1 bg-[#192540] rounded-full text-[10px] font-label uppercase text-slate-400">HuggingFace Optimized</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    if os.path.exists(DB_DIR):
        try:
            db = chromadb.PersistentClient(path=DB_DIR)
            collections = db.list_collections()
            if collections:
                collection = db.get_collection(collections[0].name)
                data = collection.get(include=['documents', 'embeddings'], limit=3)
                v_cols = st.columns(2, gap="large")
                with v_cols[0]:
                    st.markdown("""
                    <div class="bg-[#091328] rounded-xl overflow-hidden border border-white/5">
                        <div class="px-6 py-4 border-b border-white/5 flex justify-between items-center bg-[#192540]/30">
                            <span class="font-headline font-bold text-sm flex items-center gap-2">
                                <span class="material-symbols-outlined text-[#7f8af6] text-sm">segment</span>
                                Chunks ({n})
                            </span>
                        </div>
                        <div class="p-4 space-y-4">
                    """.format(n=len(data['documents'])), unsafe_allow_html=True)
                    for doc in data['documents']: 
                        st.markdown(f'<div class="p-4 bg-[#0f1930] rounded-lg border-l-4 border-[#9fa7ff] text-sm text-[#a3aac4]">"{doc[:200]}..."</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                with v_cols[1]:
                    st.markdown("""
                    <div class="bg-[#091328] rounded-xl overflow-hidden border border-white/5">
                        <div class="px-6 py-4 border-b border-white/5 flex justify-between items-center bg-[#192540]/30">
                            <span class="font-headline font-bold text-sm flex items-center gap-2">
                                <span class="material-symbols-outlined text-[#c180ff] text-sm">grid_3x3</span>
                                Embeddings (d=384)
                            </span>
                        </div>
                        <div class="p-4 space-y-4 font-label text-[10px]">
                    """, unsafe_allow_html=True)
                    for emb in data['embeddings']: 
                        st.markdown(f'<div class="p-4 bg-[#000000] rounded-lg border border-white/5 text-[#7f8af6]/80 break-all leading-tight">[{", ".join([f"{x:.2f}" for x in emb[:8]])}...]</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
        except Exception as e: st.error(f"Error: {e}")
    else: st.info("Index not found.")

elif st.session_state.active_tab == "Dashboard":
    st.markdown("""
    <div class="space-y-6">
        <h2 class="text-2xl font-extrabold font-headline tracking-tighter text-[#dee5ff]">Intelligent Query</h2>
        <div class="relative">
    """, unsafe_allow_html=True)
    
    q_in = st.text_input("Ask...", placeholder="Ask anything about the investment book...", label_visibility="collapsed")
    
    if st.button("Query AI", key="btn_q"):
        if q_in and os.path.exists(DB_DIR):
            with st.spinner("Thinking..."):
                vs = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
                chain = create_retrieval_chain(vs.as_retriever(), create_stuff_documents_chain(get_llm(), ChatPromptTemplate.from_messages([("system", "Answer with context: \n\n{context}"), ("human", "{input}")])))
                st.session_state.ans = chain.invoke({"input": q_in})
        elif not os.path.exists(DB_DIR): st.error("Upload first.")
    
    if "ans" in st.session_state:
        res = st.session_state.ans
        st.markdown(f"""
        <section class="bg-[#091328] rounded-xl p-8 lg:p-12 space-y-10 border border-white/5 mt-12">
            <div class="flex items-center gap-4 text-[#9fa7ff]">
                <span class="material-symbols-outlined">help_center</span>
                <h3 class="font-headline text-2xl font-bold">Analysis Results</h3>
            </div>
            
            <div class="space-y-4">
                <p class="font-label text-xs uppercase tracking-widest text-[#a3aac4]">Retrieved Sources (RAG Reasoning)</p>
                <div class="flex flex-col gap-3">
        """, unsafe_allow_html=True)
        
        for i, doc in enumerate(res["context"][:2]):
            st.markdown(f"""
            <div class="flex gap-4 p-4 bg-[#0f1930] rounded-lg border border-white/5">
                <span class="font-label text-[#ffa5d9] font-bold">[{i+1}]</span>
                <p class="text-sm italic text-[#a3aac4]">"{doc.page_content[:250]}..."</p>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown(f"""
                </div>
            </div>

            <div class="pt-10 border-t border-white/5 relative">
                <div class="absolute -top-4 left-0 bg-[#060e20] px-4 py-1 text-[10px] font-label font-bold text-[#9fa7ff] border border-[#9fa7ff]/20 rounded-full">FINAL ANALYSIS GENERATED</div>
                <div class="prose prose-invert max-w-none text-xl leading-relaxed text-[#dee5ff]/90">
                    {res["answer"]}
                </div>
            </div>
            
            <div class="mt-10 flex gap-4">
                <button class="flex items-center gap-2 text-sm font-bold text-[#9fa7ff] px-4 py-2 rounded-lg bg-[#9fa7ff]/10 hover:bg-[#9fa7ff]/20">
                    <span class="material-symbols-outlined text-sm">download</span> Export Analysis
                </button>
                <button class="flex items-center gap-2 text-sm font-bold text-[#a3aac4] px-4 py-2 rounded-lg hover:bg-white/5">
                    <span class="material-symbols-outlined text-sm">share</span> Share Reasoning
                </button>
            </div>
        </section>
        """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# --- GLOBAL BUTTON STYLES ---
st.markdown("""
<style>
    .stButton[key="btn_p"]>button, .stButton[key="btn_q"]>button {
        background: linear-gradient(to right, #9fa7ff, #c180ff) !important;
        color: #101b8b !important;
        font-weight: 700 !important;
        border-radius: 8px !important;
        width: 100% !important;
        padding: 1rem !important;
        border: none !important;
    }
    
    .stTextInput input {
        background-color: #000000 !important;
        border: 2px solid rgba(64, 72, 93, 0.2) !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
        color: #dee5ff !important;
        font-size: 1.25rem !important;
    }
    .stTextInput input:focus {
        border-color: #9fa7ff !important;
        box-shadow: 0 0 0 4px rgba(159, 167, 255, 0.2) !important;
    }
</style>
""", unsafe_allow_html=True)
