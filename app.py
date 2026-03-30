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
    <div class="mb-12 pt-6 px-2">
        <div class="flex items-center gap-4 mb-12">
            <div class="w-11 h-11 bg-primary rounded-2xl flex items-center justify-center text-slate-950 shadow-lg shadow-primary/20 rotate-3">
                <span class="material-symbols-outlined text-2xl" style="font-variation-settings: 'FILL' 1;">deployed_code</span>
            </div>
            <div>
                <h2 class="text-white font-headline font-bold text-lg tracking-tight">Precision</h2>
                <p class="text-[10px] text-slate-400 font-label uppercase tracking-widest leading-none">Architect v2.4</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation logic with active state styling
    cols = st.columns([1])
    with cols[0]:
        dash_active = "nav-active" if st.session_state.active_tab == "Dashboard" else ""
        if st.button("Dashboard", key="nav_dash", help="View analytics and queries", use_container_width=True): 
            st.session_state.active_tab = "Dashboard"
            st.rerun()
            
        retr_active = "nav-active" if st.session_state.active_tab == "Retriever" else ""
        if st.button("Knowledge Base", key="nav_retr", help="Upload and index documents", use_container_width=True): 
            st.session_state.active_tab = "Retriever"
            st.rerun()
            
        verify_active = "nav-active" if st.session_state.active_tab == "Verify DB" else ""
        if st.button("System Integrity", key="nav_verify", help="Check vector database health", use_container_width=True): 
            st.session_state.active_tab = "Verify DB"
            st.rerun()

    # Injecting active state CSS for specific buttons
    st.markdown(f"""
    <style>
        div[data-testid="stButton"] button[key="nav_dash"] {{ {f'background-color: rgba(94, 102, 255, 0.15) !important; color: white !important; border-left: 3px solid #5e66ff !important;' if st.session_state.active_tab == "Dashboard" else ''} }}
        div[data-testid="stButton"] button[key="nav_retr"] {{ {f'background-color: rgba(94, 102, 255, 0.15) !important; color: white !important; border-left: 3px solid #5e66ff !important;' if st.session_state.active_tab == "Retriever" else ''} }}
        div[data-testid="stButton"] button[key="nav_verify"] {{ {f'background-color: rgba(94, 102, 255, 0.15) !important; color: white !important; border-left: 3px solid #5e66ff !important;' if st.session_state.active_tab == "Verify DB" else ''} }}
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="mt-auto mb-8 pt-6 px-2 border-t border-white/5">
        <div class="flex items-center gap-3 text-slate-400 text-xs px-2 py-3 bg-slate-800/40 rounded-xl">
            <span class="relative flex h-2 w-2">
                <span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                <span class="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
            </span>
            <span class="font-medium">System Engine: Optimal</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- CUSTOM HEADER ---
st.markdown(f"""
<div class="flex items-center justify-between px-10 py-6 bg-slate-950/50 backdrop-blur-md border-b border-white/5 sticky top-0 z-50">
    <div class="flex items-center gap-10">
        <span class="text-xl font-headline font-extrabold tracking-tighter bg-gradient-to-r from-white to-slate-400 bg-clip-text text-transparent">RAG ANALYZER</span>
        <div class="hidden md:flex gap-8">
            <span class="text-sm font-medium text-slate-400 hover:text-white transition-colors cursor-pointer">Portfolio</span>
            <span class="text-sm font-medium text-primary border-b-2 border-primary pb-1">Intelligence</span>
            <span class="text-sm font-medium text-slate-400 hover:text-white transition-colors cursor-pointer">Resources</span>
        </div>
    </div>
    <div class="flex items-center gap-6">
        <div class="flex items-center gap-3 bg-slate-900 px-4 py-2 rounded-2xl border border-white/5">
            <span class="relative flex h-2 w-2">
                <span class="animate-pulse absolute inline-flex h-full w-full rounded-full bg-primary-400 opacity-75"></span>
                <span class="relative inline-flex rounded-full h-2 w-2 bg-primary"></span>
            </span>
            <span class="text-[10px] uppercase tracking-widest text-primary-300 font-bold font-label">{st.session_state.process_status}</span>
        </div>
        <div class="w-10 h-10 rounded-full border-2 border-primary/20 p-0.5">
            <img src="https://lh3.googleusercontent.com/aida-public/AB6AXuCHg42IWLey4S15SqkykRxyXZ74kddwgq21vEX9uiG7QWnY_jlHlpQHBelnSrAMrvANaeXGQBJZ2qZn0f6kOOvLsoq7Q-cq379cmg6TdqL4rT_0XzNRWi4dIwrFilyDGKhr5yCyohoXL1xviVF5bUuRt8Yifxe-VxcnY4D8YLCtxxYVNw8POr7pS-0JH8PXge7dq7HkrecSoLkD56oNshJ_iv4w7kMyv3riQuGn5LZXVdB1-NXyu6BOa6vvncteixXZfnbw3ewfQKE" class="w-full h-full rounded-full object-cover shadow-lg">
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- MAIN CONTENT ---
st.markdown('<div class="main-content">', unsafe_allow_html=True)

if st.session_state.active_tab == "Retriever":
    st.markdown("""
    <div class="space-y-12">
        <div class="flex items-end justify-between">
            <div class="space-y-3">
                <div class="flex items-center gap-3">
                    <span class="px-3 py-1 bg-primary/10 text-primary text-[10px] font-bold tracking-[0.2em] rounded-full uppercase">Ingestion Engine</span>
                </div>
                <h1 class="text-5xl font-black font-headline tracking-tight text-white leading-tight">Knowledge Base</h1>
                <p class="text-slate-400 text-lg max-w-2xl font-medium">Initialize your analysis by processing financial literature and institutional reports into the neural vector engine.</p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    r_cols = st.columns([1.8, 1], gap="large")
    with r_cols[0]:
        st.markdown("""
        <div class="glass-card p-12 flex flex-col items-center justify-center text-center space-y-8 relative overflow-hidden group">
            <div class="absolute inset-0 bg-gradient-to-br from-primary/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500"></div>
            <div class="w-24 h-24 bg-slate-900 rounded-[32px] flex items-center justify-center text-primary-400 border border-white/5 shadow-2xl group-hover:scale-110 group-hover:rotate-3 transition-all duration-500">
                <span class="material-symbols-outlined text-5xl" style="font-variation-settings: 'FILL' 1;">cloud_upload</span>
            </div>
            <div class="space-y-3">
                <h3 class="text-2xl font-bold font-headline text-white tracking-tight">Drop investment documents here</h3>
                <p class="text-slate-400 font-medium">Maximum file size: <span class="text-white">50MB</span> • Supports <span class="text-white">PDF, DOCX</span></p>
            </div>
            <div class="px-8 py-3 bg-white/5 rounded-2xl border border-white/10 group-hover:border-primary/30 transition-all font-semibold text-slate-300">
                Browse Files
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Hidden file uploader layered on top or styled
        uploaded_file = st.file_uploader("Upload", type="pdf", label_visibility="collapsed")

    with r_cols[1]:
        # Build Document Status HTML
        status_text = "Ready for Ingestion" if uploaded_file else "Awaiting Selection"
        file_name = uploaded_file.name if uploaded_file else "-"
        file_size = (uploaded_file.size/1024/1024) if uploaded_file else 0
        
        st.markdown(f"""
        <div class="glass-card p-8 flex flex-col h-full border-l-4 border-l-primary/40">
            <div class="flex-grow space-y-8">
                <div class="flex items-center gap-5">
                    <div class="w-14 h-14 bg-rose-500/10 text-rose-400 rounded-2xl flex items-center justify-center shadow-inner border border-rose-500/20">
                        <span class="material-symbols-outlined text-3xl">description</span>
                    </div>
                    <div>
                        <p class="text-xs font-bold text-slate-500 uppercase tracking-widest mb-1">Active Artifact</p>
                        <p class="font-bold text-white text-lg truncate w-48 font-headline">{file_name}</p>
                    </div>
                </div>
                
                <div class="space-y-4">
                    <div class="flex justify-between items-end">
                        <div class="space-y-1">
                            <p class="text-[10px] font-bold text-slate-500 uppercase tracking-widest">Metadata</p>
                            <p class="text-sm font-semibold text-slate-300">{file_size:.2f} MB • PDF 1.7</p>
                        </div>
                        <div class="text-right space-y-1">
                            <p class="text-[10px] font-bold text-slate-500 uppercase tracking-widest">Status</p>
                            <p class="text-sm font-bold text-primary-400">{status_text}</p>
                        </div>
                    </div>
                    <div class="w-full h-2 bg-slate-900 rounded-full overflow-hidden border border-white/5">
                        <div class="h-full bg-gradient-to-r from-primary-600 to-primary-400 w-full opacity-30"></div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        if uploaded_file:
            if st.button("Begin Neural Indexing", key="btn_p", use_container_width=True):
                with st.spinner("Analyzing Document Topology..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.getvalue()); tmp_path = tmp.name
                    try:
                        loader = PyPDFLoader(tmp_path); docs = loader.load()
                        splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
                        Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=DB_DIR)
                        st.session_state.process_status = "STABLE"; st.success("Knowledge Ingested Successfully")
                        st.rerun()
                    finally: os.unlink(tmp_path)
        
        st.markdown("""
            <div class="mt-8 pt-6 border-t border-white/5 flex items-center gap-3 opacity-60">
                <span class="material-symbols-outlined text-sm text-slate-400">info</span>
                <p class="text-[10px] text-slate-400 italic">Vectorization uses HuggingFace all-MiniLM-L6-v2</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True) # Closes space-y-12

elif st.session_state.active_tab == "Verify DB":
    st.markdown("""
    <div class="space-y-10">
        <div class="flex items-end justify-between">
            <div class="space-y-3">
                <div class="flex items-center gap-3">
                    <span class="px-3 py-1 bg-primary/10 text-primary text-[10px] font-bold tracking-[0.2em] rounded-full uppercase">Database Integrity</span>
                </div>
                <h1 class="text-5xl font-black font-headline tracking-tight text-white leading-tight">System Analytics</h1>
                <p class="text-slate-400 text-lg max-w-2xl font-medium">Direct inspection of the latent document embeddings and partitioned knowledge chunks.</p>
            </div>
            <div class="flex gap-3">
                <div class="px-4 py-2 bg-slate-900 rounded-xl border border-white/5 text-[10px] font-bold text-slate-500 uppercase tracking-widest">HuggingFace Optimized</div>
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
                    for i, doc in enumerate(data['documents']):
                        chunks_html += f"""
                        <div class="p-5 bg-slate-900/50 rounded-2xl border border-white/5 space-y-3 group hover:border-primary/30 transition-all">
                            <div class="flex items-center gap-2">
                                <span class="w-2 h-2 rounded-full bg-primary opacity-40 group-hover:opacity-100 transition-opacity"></span>
                                <span class="text-[10px] font-black text-slate-500 uppercase tracking-widest">Chunk {i+1}</span>
                            </div>
                            <p class="text-sm text-slate-400 italic leading-relaxed">"{doc[:240]}..."</p>
                        </div>
                        """
                    
                    st.markdown(f"""
                    <div class="glass-card overflow-hidden">
                        <div class="px-8 py-5 border-b border-white/5 flex justify-between items-center bg-white/5">
                            <span class="font-headline font-bold text-lg text-white flex items-center gap-3">
                                <span class="material-symbols-outlined text-primary">segment</span>
                                Knowledge Segments
                            </span>
                            <span class="px-3 py-1 bg-primary/10 text-primary text-[10px] font-bold rounded-lg uppercase tracking-widest">{len(data['documents'])} Nodes</span>
                        </div>
                        <div class="p-8 space-y-6">
                            {chunks_html}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with v_cols[1]:
                    embeddings_html = ""
                    for i, emb in enumerate(data['embeddings']):
                        embeddings_html += f"""
                        <div class="p-5 bg-slate-950 rounded-2xl border border-white/5 space-y-3 group hover:border-primary/30 transition-all">
                            <div class="flex items-center gap-2">
                                <span class="w-2 h-2 rounded-full bg-primary-400 opacity-40 group-hover:opacity-100 transition-opacity"></span>
                                <span class="text-[10px] font-black text-slate-500 uppercase tracking-widest">Embedding {i+1}</span>
                            </div>
                            <p class="text-[9px] font-mono text-primary/60 break-all leading-tight">[{", ".join([f"{x:.3f}" for x in emb[:15]])}...]</p>
                        </div>
                        """
                    
                    st.markdown(f"""
                    <div class="glass-card overflow-hidden">
                        <div class="px-8 py-5 border-b border-white/5 flex justify-between items-center bg-white/5">
                            <span class="font-headline font-bold text-lg text-white flex items-center gap-3">
                                <span class="material-symbols-outlined text-primary">analytics</span>
                                Vector Embeddings
                            </span>
                            <span class="px-3 py-1 bg-primary/10 text-primary text-[10px] font-bold rounded-lg uppercase tracking-widest">d=384</span>
                        </div>
                        <div class="p-8 space-y-6">
                            {embeddings_html}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        except Exception as e: st.error(f"Integrity check failed: {e}")
    else: st.info("Intelligence core is offline. Please ingest documents to initialize the vector database.")
    st.markdown("</div>", unsafe_allow_html=True)

elif st.session_state.active_tab == "Dashboard":
    st.markdown("""
    <div class="space-y-10">
        <div class="space-y-3">
            <div class="flex items-center gap-3">
                <span class="px-3 py-1 bg-primary/10 text-primary text-[10px] font-bold tracking-[0.2em] rounded-full uppercase">Cognitive Layer</span>
            </div>
            <h1 class="text-5xl font-black font-headline tracking-tight text-white leading-tight">Intelligent Query</h1>
            <p class="text-slate-400 text-lg max-w-2xl font-medium">Probe the latent space of your ingested documents with precise, natural language inquiries.</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Custom Query Input Styling
    st.markdown("""
    <div class="relative group mt-8">
        <div class="absolute inset-0 bg-primary/20 blur-2xl rounded-full opacity-0 group-focus-within:opacity-100 transition-opacity duration-500"></div>
    </div>
    """, unsafe_allow_html=True)
    
    q_in = st.text_input("Ask...", placeholder="What are the key risk factors mentioned in the portfolio report?", label_visibility="collapsed")
    
    q_cols = st.columns([1, 4, 1])
    with q_cols[1]:
        if st.button("Execute Neural Search", key="btn_q", use_container_width=True):
            if q_in and os.path.exists(DB_DIR):
                with st.spinner("Synthesizing Insights..."):
                    vs = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
                    chain = create_retrieval_chain(vs.as_retriever(), create_stuff_documents_chain(get_llm(), ChatPromptTemplate.from_messages([("system", "Answer with context: \n\n{context}"), ("human", "{input}")])))
                    st.session_state.ans = chain.invoke({"input": q_in})
            elif not os.path.exists(DB_DIR): st.error("No knowledge base found. Please ingest documents first.")
    
    if "ans" in st.session_state:
        res = st.session_state.ans
        
        # Build Sources HTML
        sources_html = ""
        for i, doc in enumerate(res["context"][:2]):
            sources_html += f"""
            <div class="glass-card p-6 border-l-2 border-l-primary/30 hover:border-l-primary transition-all">
                <div class="flex items-center gap-3 mb-3">
                    <span class="text-[10px] font-black text-primary uppercase tracking-[0.2em]">Context node {i+1}</span>
                    <div class="h-px flex-grow bg-white/5"></div>
                </div>
                <p class="text-sm italic text-slate-400 leading-relaxed font-medium">"{doc.page_content[:320]}..."</p>
            </div>
            """

        # Full Results Block
        st.markdown(f"""
        <div class="glass-card p-10 lg:p-16 space-y-16 mt-16 relative overflow-hidden">
            <div class="absolute top-0 right-0 w-64 h-64 bg-primary/5 blur-[100px] -z-10"></div>
            
            <div class="flex flex-col md:flex-row md:items-center justify-between gap-8">
                <div class="flex items-center gap-5">
                    <div class="w-16 h-16 bg-primary rounded-3xl flex items-center justify-center text-slate-950 shadow-2xl shadow-primary/20 rotate-6">
                        <span class="material-symbols-outlined text-3xl" style="font-variation-settings: 'FILL' 1;">insights</span>
                    </div>
                    <div>
                        <h3 class="font-headline text-3xl font-black text-white tracking-tight">Synthesis Report</h3>
                        <p class="text-slate-400 font-medium">Generated from multidimensional vector retrieval</p>
                    </div>
                </div>
                <div class="flex gap-3">
                    <div class="px-4 py-2 bg-slate-900 rounded-xl border border-white/5 text-[10px] font-bold text-slate-500 uppercase tracking-widest">v2.4 STABLE</div>
                </div>
            </div>
            
            <div class="space-y-8">
                <div class="flex items-center gap-4">
                    <p class="font-label text-[10px] uppercase tracking-[0.4em] text-primary font-black whitespace-nowrap">Retrieved Context</p>
                    <div class="h-px w-full bg-gradient-to-r from-primary/20 to-transparent"></div>
                </div>
                <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                    {sources_html}
                </div>
            </div>

            <div class="pt-16 border-t border-white/5 relative">
                <div class="absolute -top-5 left-0 bg-slate-950 px-6 py-2 text-[10px] font-black text-primary border border-primary/30 rounded-full flex items-center gap-3">
                    <span class="relative flex h-2 w-2">
                        <span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-primary opacity-75"></span>
                        <span class="relative inline-flex rounded-full h-2 w-2 bg-primary"></span>
                    </span>
                    ANALYSIS FINALIZED
                </div>
                <div class="prose prose-invert max-w-none">
                    <div class="text-xl leading-relaxed text-slate-200 font-medium font-body opacity-95">
                        {res["answer"]}
                    </div>
                </div>
            </div>
            
            <div class="mt-16 flex flex-wrap gap-4 pt-10 border-t border-white/5">
                <button class="bg-primary hover:bg-primary-600 text-slate-950 font-bold px-8 py-4 rounded-2xl transition-all hover:scale-105 active:scale-95 shadow-xl shadow-primary/10 flex items-center gap-3">
                    <span class="material-symbols-outlined text-xl">file_download</span> 
                    <span>Export Analysis</span>
                </button>
                <button class="bg-white/5 hover:bg-white/10 text-white font-bold px-8 py-4 rounded-2xl border border-white/10 transition-all active:scale-95 flex items-center gap-3">
                    <span class="material-symbols-outlined text-xl">share</span> 
                    <span>Share Insights</span>
                </button>
            </div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True) # Closes space-y-10

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
