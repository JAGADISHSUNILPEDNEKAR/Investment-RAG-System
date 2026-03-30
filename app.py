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
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap" rel="stylesheet"/>
<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200" rel="stylesheet"/>
<script>
tailwind.config = {
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        primary: { 
          50: '#f0f2ff', 100: '#e0e5ff', 200: '#c1ccff', 300: '#9faaff', 400: '#7d88ff', 
          500: '#5e66ff', 600: '#4b52cc', 700: '#383d99', 800: '#262966', 900: '#131533',
          DEFAULT: '#5e66ff'
        },
        slate: { 
          950: '#060e20', 900: '#091328', 800: '#0f1930', 700: '#192540', 600: '#1f2b49' 
        }
      },
      fontFamily: {
        headline: ["Outfit", "sans-serif"],
        body: ["Inter", "sans-serif"],
        label: ["Space Grotesk", "sans-serif"]
      },
      animation: {
        'pulse-slow': 'pulse 4s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'float': 'float 3s ease-in-out infinite',
      },
      keyframes: {
        float: {
          '0%, 100%': { transform: 'translateY(0)' },
          '50%': { transform: 'translateY(-10px)' },
        }
      }
    }
  }
};
</script>
<style>
    :root {
        --glass-bg: rgba(15, 25, 48, 0.7);
        --glass-border: rgba(255, 255, 255, 0.08);
        --sidebar-width: 280px;
        --content-gap: 80px;
    }

    .stApp {
        background-color: #060e20;
        color: #dee5ff;
        font-family: 'Inter', sans-serif;
    }
    
    .block-container {
        padding: 0 !important;
        max-width: 100% !important;
    }
    
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }

    /* Docked Sidebar */
    [data-testid="stSidebar"] {
        background-color: #091328 !important;
        border-right: 1px solid var(--glass-border);
        width: var(--sidebar-width) !important;
        box-shadow: 10px 0 30px rgba(0,0,0,0.5);
    }
    [data-testid="stSidebarNav"] { display: none !important; }

    /* Main Content Wrapper - Fixing Spacing */
    .main-content {
        padding: 4rem var(--content-gap) 4rem calc(var(--content-gap) + 20px);
        max-width: 1600px;
        margin: 0 auto;
        animation: fadeIn 0.8s ease-out;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Glass Cards */
    .glass-card {
        background: var(--glass-bg);
        backdrop-filter: blur(12px);
        border: 1px solid var(--glass-border);
        border-radius: 24px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .glass-card:hover {
        border-color: rgba(94, 102, 255, 0.3);
        box-shadow: 0 20px 40px rgba(0,0,0,0.4);
        transform: translateY(-2px);
    }

    /* Sidebar Buttons */
    section[data-testid="stSidebar"] .stButton>button {
        background-color: transparent !important;
        color: #a3aac4 !important;
        border: 1px solid transparent !important;
        text-align: left !important;
        padding: 0.85rem 1.25rem !important;
        width: 100% !important;
        display: flex !important;
        align-items: center !important;
        gap: 14px !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 500 !important;
        border-radius: 12px !important;
        transition: all 0.2s !important;
    }
    section[data-testid="stSidebar"] .stButton>button:hover {
        background-color: rgba(94, 102, 255, 0.1) !important;
        color: #fff !important;
        border-color: rgba(94, 102, 255, 0.2) !important;
    }
    
    /* Active Nav Styling */
    .nav-active {
        background-color: rgba(94, 102, 255, 0.15) !important;
        color: #fff !important;
        border-left: 3px solid #5e66ff !important;
    }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: #060e20; }
    ::-webkit-scrollbar-thumb { background: #1f2b49; border-radius: 10px; border: 2px solid #060e20; }
    ::-webkit-scrollbar-thumb:hover { background: #5e66ff; }
</style>
""", unsafe_allow_html=True)

# --- BACKEND ---
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@st.cache_resource
def get_llm():
    return ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0.2)

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
        # Build Document Status HTML
        status_text = "File selected" if uploaded_file else "No file selected"
        file_size = (uploaded_file.size/1024/1024) if uploaded_file else 0
        
        st.markdown(f"""
        <div class="bg-[#0f1930] p-8 rounded-xl flex flex-col justify-between h-full border border-white/5">
            <div class="space-y-6">
                <div class="flex items-start justify-between">
                    <div class="flex items-center gap-4">
                        <div class="w-12 h-12 bg-[#a70138]/20 text-[#ff6e84] rounded-xl flex items-center justify-center shadow-lg">
                            <span class="material-symbols-outlined text-2xl">picture_as_pdf</span>
                        </div>
                        <div>
                            <p class="font-bold text-[#dee5ff] text-base truncate w-40">{status_text}</p>
                            <p class="text-[10px] font-label text-slate-500 uppercase tracking-widest">{file_size:.1f} MB • PDF ARCHIVE</p>
                        </div>
                    </div>
                </div>
                
                <div class="space-y-2">
                    <div class="flex justify-between text-[10px] font-label text-slate-400">
                        <span>PROCESSING ENGINE</span>
                        <span>v2.4 STABLE</span>
                    </div>
                    <div class="w-full h-1 bg-white/5 rounded-full overflow-hidden">
                        <div class="h-full bg-[#9fa7ff] w-1/3 opacity-50"></div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        if uploaded_file and st.button("Initialize Neural Index", key="btn_p"):
            with st.spinner("Analyzing Document Topology..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.getvalue()); tmp_path = tmp.name
                try:
                    loader = PyPDFLoader(tmp_path); docs = loader.load()
                    splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
                    Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=DB_DIR)
                    st.session_state.process_status = "READY"; st.success("Neural Indexing Complete")
                finally: os.unlink(tmp_path)
        
        st.markdown("""
            <p class="text-[10px] text-[#a3aac4] mt-6 italic text-center opacity-60">Click above to start RAG indexing</p>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True) # Closes space-y-12

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
                    chunks_html = ""
                    for doc in data['documents']:
                        chunks_html += f'<div class="p-4 bg-[#0f1930] rounded-lg border-l-4 border-[#9fa7ff] text-sm text-[#a3aac4] shadow-sm italic leading-relaxed">"{doc[:220]}..."</div>'
                    
                    st.markdown(f"""
                    <div class="bg-[#091328] rounded-xl overflow-hidden border border-white/5 shadow-xl">
                        <div class="px-6 py-4 border-b border-white/5 flex justify-between items-center bg-[#192540]/30">
                            <span class="font-headline font-bold text-sm flex items-center gap-3">
                                <span class="material-symbols-outlined text-[#7f8af6]">segment</span>
                                Document Chunks ({len(data['documents'])})
                            </span>
                        </div>
                        <div class="p-5 space-y-4">
                            {chunks_html}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with v_cols[1]:
                    embeddings_html = ""
                    for emb in data['embeddings']:
                        embeddings_html += f'<div class="p-4 bg-[#000000] rounded-lg border border-white/5 text-[#7f8af6]/80 break-all leading-tight font-mono text-[9px] shadow-inner">[{", ".join([f"{x:.2f}" for x in emb[:12]])}...]</div>'
                    
                    st.markdown(f"""
                    <div class="bg-[#091328] rounded-xl overflow-hidden border border-white/5 shadow-xl">
                        <div class="px-6 py-4 border-b border-white/5 flex justify-between items-center bg-[#192540]/30">
                            <span class="font-headline font-bold text-sm flex items-center gap-3">
                                <span class="material-symbols-outlined text-[#c180ff]">grid_3x3</span>
                                High-Dim Vectors (d=384)
                            </span>
                        </div>
                        <div class="p-5 space-y-4">
                            {embeddings_html}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        except Exception as e: st.error(f"Error: {e}")
    else: st.info("Index not found.")

elif st.session_state.active_tab == "Dashboard":
    st.markdown("""
    <div class="space-y-8">
        <div class="space-y-2">
            <h2 class="text-3xl font-extrabold font-headline tracking-tighter text-[#dee5ff]">Intelligent Query</h2>
            <p class="text-[#a3aac4] text-lg">Probe the cognitive architecture of the investment literature with precise inquiries.</p>
        </div>
    """, unsafe_allow_html=True)
    
    q_in = st.text_input("Ask...", placeholder="Ask anything about the investment documents...", label_visibility="collapsed")
    
    if st.button("Query AI", key="btn_q"):
        if q_in and os.path.exists(DB_DIR):
            with st.spinner("Thinking..."):
                vs = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
                chain = create_retrieval_chain(vs.as_retriever(), create_stuff_documents_chain(get_llm(), ChatPromptTemplate.from_messages([("system", "Answer with context: \n\n{context}"), ("human", "{input}")])))
                st.session_state.ans = chain.invoke({"input": q_in})
        elif not os.path.exists(DB_DIR): st.error("Upload first.")
    
    if "ans" in st.session_state:
        res = st.session_state.ans
        
        # Build Sources HTML
        sources_html = ""
        for i, doc in enumerate(res["context"][:2]):
            sources_html += f"""
            <div class="flex gap-4 p-5 bg-[#192540]/40 rounded-xl border border-white/5 hover:border-[#9fa7ff]/30 transition-all">
                <span class="font-label text-[#ffa5d9] font-bold text-sm">SOURCE [{i+1}]</span>
                <p class="text-sm italic text-[#a3aac4] leading-relaxed">"{doc.page_content[:280]}..."</p>
            </div>
            """

        # Full Results Block
        st.markdown(f"""
        <section class="bg-[#091328] rounded-2xl p-8 lg:p-12 space-y-12 border border-white/5 mt-12 shadow-2xl">
            <div class="flex items-center gap-4 text-[#9fa7ff]">
                <div class="w-12 h-12 bg-[#9fa7ff]/10 rounded-full flex items-center justify-center">
                    <span class="material-symbols-outlined text-2xl">analytics</span>
                </div>
                <h3 class="font-headline text-3xl font-bold tracking-tight">Synthesis Report</h3>
            </div>
            
            <div class="space-y-6">
                <div class="flex items-center gap-3">
                    <div class="h-[1px] flex-grow bg-gradient-to-r from-transparent via-white/10 to-transparent"></div>
                    <p class="font-label text-[10px] uppercase tracking-[0.3em] text-[#a3aac4] font-bold">Retrieved Reasoning Nodes</p>
                    <div class="h-[1px] flex-grow bg-gradient-to-r from-transparent via-white/10 to-transparent"></div>
                </div>
                <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {sources_html}
                </div>
            </div>

            <div class="pt-12 border-t border-white/5 relative">
                <div class="absolute -top-4 left-0 bg-[#060e20] px-5 py-1.5 text-[10px] font-label font-bold text-[#9fa7ff] border border-[#9fa7ff]/30 rounded-full flex items-center gap-2">
                    <span class="relative flex h-2 w-2">
                        <span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-[#9fa7ff] opacity-75"></span>
                        <span class="relative inline-flex rounded-full h-2 w-2 bg-[#9fa7ff]"></span>
                    </span>
                    FINAL ANALYSIS GENERATED
                </div>
                <div class="prose prose-invert max-w-none text-xl leading-relaxed text-[#dee5ff]/90 font-body">
                    {res["answer"]}
                </div>
            </div>
            
            <div class="mt-12 flex flex-wrap gap-4 pt-6">
                <button class="flex items-center gap-3 text-sm font-bold text-[#101b8b] px-6 py-3 rounded-xl bg-[#9fa7ff] hover:bg-[#8d98ff] transition-all transform active:scale-95 shadow-lg shadow-[#9fa7ff]/10">
                    <span class="material-symbols-outlined text-sm">file_download</span> Export Full Analysis
                </button>
                <button class="flex items-center gap-3 text-sm font-bold text-[#a3aac4] px-6 py-3 rounded-xl bg-white/5 hover:bg-white/10 border border-white/5 transition-all active:scale-95">
                    <span class="material-symbols-outlined text-sm">share</span> Share Core Insights
                </button>
            </div>
        </section>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True) # Closes space-y-8

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
