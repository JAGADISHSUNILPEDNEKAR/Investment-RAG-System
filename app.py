import streamlit as st
import os
import tempfile
import textwrap
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

# --- DARK THEME TAILWIND & CSS ---
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
        --content-gap: 120px;
    }

    .stApp {
        background-color: #060e20;
        color: #dee5ff;
        font-family: 'Inter', sans-serif;
    }

    div[data-testid="stAppViewContainer"] section.main div.block-container {
        padding: 90px 4rem 4rem 312px !important;
        max-width: none !important;
        margin-left: 0 !important;
        margin-right: auto !important;
    }

    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }

    /* Docked Sidebar */
    [data-testid="stSidebar"] {
        background-color: #091328 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.12) !important;
        width: var(--sidebar-width) !important;
        box-shadow: 20px 0 60px rgba(0,0,0,0.6);
    }
    [data-testid="stSidebarNav"] { display: none !important; }

    /* Main Content Wrapper - using more intentional spacing */
    .stMain {
        background-color: #060e20;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Glass Cards */
    .glass-card {
        background: rgba(15, 25, 48, 0.4);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 24px;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .glass-card:hover {
        border-color: rgba(94, 102, 255, 0.3);
        box-shadow: 0 20px 40px rgba(0,0,0,0.4);
        transform: translateY(-2px);
    }

    /* Sidebar Buttons - base style */
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

    /* Scrollbar */
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: #060e20; }
    ::-webkit-scrollbar-thumb { background: #1f2b49; border-radius: 10px; border: 2px solid #060e20; }
    ::-webkit-scrollbar-thumb:hover { background: #5e66ff; }

    /* Primary Action Buttons */
    .stButton>button[kind="primary"] {
        background: #5e66ff !important;
        color: #060e20 !important;
        padding: 1rem !important;
        border: none !important;
        border-radius: 16px !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 600 !important;
        box-shadow: 0 10px 20px rgba(94, 102, 255, 0.2) !important;
        transition: all 0.3s !important;
    }
    .stButton>button[kind="primary"]:hover {
        background: #7d88ff !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 15px 30px rgba(94, 102, 255, 0.3) !important;
    }

    /* Text Inputs */
    .stTextInput input {
        background-color: #091328 !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 20px !important;
        padding: 1.5rem 2rem !important;
        color: #fff !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 1.1rem !important;
        transition: all 0.3s !important;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.2) !important;
    }
    .stTextInput input:focus {
        border-color: #5e66ff !important;
        background-color: #0f1930 !important;
        box-shadow: 0 0 0 4px rgba(94, 102, 255, 0.1) !important;
    }

    /* Success/Error override */
    .stSuccess, .stError, .stInfo {
        background: rgba(15, 25, 48, 0.8) !important;
        backdrop-filter: blur(10px) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 16px !important;
        color: #dee5ff !important;
    }

    /* Fixed Modern Header */
    .precision-header {
        position: fixed !important;
        top: 0 !important;
        right: 0 !important;
        left: 280px !important; /* Sidebar width from line 70 */
        height: 80px !important;
        background: rgba(6, 14, 32, 0.6) !important;
        backdrop-filter: blur(24px) saturate(180%) !important;
        -webkit-backdrop-filter: blur(24px) saturate(180%) !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08) !important;
        z-index: 999991 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: space-between !important;
        padding: 0 32px !important; /* 32px horizontal padding */
        transition: all 0.3s ease !important;
    }

    @media (max-width: 768px) {
        .precision-header { left: 0 !important; padding: 0 1.5rem !important; }
        .block-container { padding: 120px 1.5rem 4rem 1.5rem !important; }
    }
</style>
""", unsafe_allow_html=True)

# --- BACKEND ---
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# FIX #5: Native Streamlit Secrets Handling with fallback to .env for local dev
def get_google_api_key():
    if "GOOGLE_API_KEY" in st.secrets:
        return st.secrets["GOOGLE_API_KEY"]
    return os.getenv("GOOGLE_API_KEY")

@st.cache_resource
def get_llm():
    api_key = get_google_api_key()
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        temperature=0.2,
        google_api_key=api_key
    )

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

    if st.button("Dashboard", key="nav_dash", use_container_width=True):
        st.session_state.active_tab = "Dashboard"
        st.rerun()

    if st.button("Knowledge Base", key="nav_retr", use_container_width=True):
        st.session_state.active_tab = "Retriever"
        st.rerun()

    if st.button("System Integrity", key="nav_verify", use_container_width=True):
        st.session_state.active_tab = "Verify DB"
        st.rerun()

    # FIX #3 & #7: Use JS targeting by button text content — the only reliable
    # approach since Streamlit does not expose the Python `key` as an HTML attribute.
    tab_label_map = {
        "Dashboard": "Dashboard",
        "Retriever": "Knowledge Base",
        "Verify DB": "System Integrity",
    }
    active_label = tab_label_map.get(st.session_state.active_tab, "")
    st.markdown(f"""
    <script>
    (function applyActiveNav() {{
        const activeLabel = {active_label!r};
        const buttons = window.parent.document.querySelectorAll(
            'section[data-testid="stSidebar"] button'
        );
        buttons.forEach(btn => {{
            const label = btn.innerText.trim();
            if (label === activeLabel) {{
                btn.style.backgroundColor = 'rgba(94, 102, 255, 0.15)';
                btn.style.color = '#ffffff';
                btn.style.borderLeft = '3px solid #5e66ff';
            }} else {{
                btn.style.backgroundColor = '';
                btn.style.color = '';
                btn.style.borderLeft = '';
            }}
        }});
    }})();
    // Re-run after Streamlit re-renders the DOM
    setTimeout(() => {{
        const activeLabel = {active_label!r};
        const buttons = window.parent.document.querySelectorAll(
            'section[data-testid="stSidebar"] button'
        );
        buttons.forEach(btn => {{
            const label = btn.innerText.trim();
            if (label === activeLabel) {{
                btn.style.backgroundColor = 'rgba(94, 102, 255, 0.15)';
                btn.style.color = '#ffffff';
                btn.style.borderLeft = '3px solid #5e66ff';
            }} else {{
                btn.style.backgroundColor = '';
                btn.style.color = '';
                btn.style.borderLeft = '';
            }}
        }});
    }}, 300);
    </script>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="mt-auto mb-8 pt-6 px-2 border-t border-white/5">
        <div class="flex items-center gap-3 text-slate-400 text-xs px-2 py-3 bg-slate-800/40 rounded-xl">
            <span class="relative flex h-2 w-2">
                <span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                <span class="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
            </span>
            <span class="font-medium">System Engine: {"Optimal" if get_google_api_key() else "Offline"}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- FIXED MODERN HEADER ---
st.markdown(f"""
<div class="precision-header">
    <div class="flex items-center gap-6">
        <span class="text-xl font-headline font-black tracking-tighter bg-gradient-to-r from-white to-slate-400 bg-clip-text text-transparent">PRECISION AI</span>
        <div class="hidden lg:flex items-center gap-4 border-l border-white/10 pl-6 h-6">
            <span class="text-[10px] font-bold text-slate-500 uppercase tracking-widest leading-none">Intelligence Engine</span>
        </div>
    </div>
    <div class="flex items-center gap-6">
        <div class="flex items-center gap-2.5 bg-white/5 px-4 py-1.5 rounded-full border border-white/10 shadow-inner">
            <span class="relative flex h-2 w-2">
                <span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                <span class="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
            </span>
            <span class="text-[10px] uppercase font-bold tracking-widest text-emerald-400/90 leading-none">{st.session_state.process_status}</span>
        </div>
        <div class="rounded-full border border-white/20 p-0.5 hover:border-primary/50 hover:scale-105 transition-all cursor-pointer ring-4 ring-black/20" style="width: 40px; height: 40px; overflow: hidden;">
            <img src="https://lh3.googleusercontent.com/aida-public/AB6AXuCHg42IWLey4S15SqkykRxyXZ74kddwgq21vEX9uiG7QWnY_jlHlpQHBelnSrAMrvANaeXGQBJZ2qZn0f6kOOvLsoq7Q-cq379cmg6TdqL4rT_0XzNRWi4dIwrFilyDGKhr5yCyohoXL1xviVF5bUuRt8Yifxe-VxcnY4D8YLCtxxYVNw8POr7pS-0JH8PXge7dq7HkrecSoLkD56oNshJ_iv4w7kMyv3riQuGn5LZXVdB1-NXyu6BOa6vvncteixXZfnbw3ewfQKE" style="width: 100%; height: 100%; object-fit: cover;">
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- MAIN CONTENT ---
# (Removed the broken main-content div wrapper)

# ==============================================================================
# TAB: KNOWLEDGE BASE (Retriever)
# ==============================================================================
if st.session_state.active_tab == "Retriever":
    st.markdown("""
    <div class="space-y-16 mt-10">
        <div class="flex items-end justify-between border-b border-white/5 pb-12">
            <div class="space-y-5">
                <div class="flex items-center gap-3">
                    <div class="flex items-center gap-2 px-3 py-1 bg-primary/10 border border-primary/20 rounded-full">
                        <span class="w-1.5 h-1.5 rounded-full bg-primary animate-pulse"></span>
                        <span class="text-primary text-[9px] font-black tracking-[0.25em] uppercase">Ingestion Engine</span>
                    </div>
                </div>
                <h1 class="text-6xl font-black font-headline tracking-tighter text-white leading-none">Knowledge Base</h1>
                <p class="text-slate-400 text-lg max-w-2xl font-medium leading-relaxed">Initialize your analysis by processing financial literature and institutional reports into the neural vector engine.</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    r_cols = st.columns([1.8, 1], gap="large")

    with r_cols[0]:
        st.markdown(textwrap.dedent("""
        <div class="glass-card p-12 flex flex-col items-center justify-center text-center space-y-8 relative overflow-hidden group">
            <div class="absolute inset-0 bg-gradient-to-br from-primary/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500"></div>
            <div class="w-24 h-24 bg-slate-900 rounded-[32px] flex items-center justify-center text-primary-400 border border-white/5 shadow-2xl group-hover:scale-110 group-hover:rotate-3 transition-all duration-500">
                <span class="material-symbols-outlined text-5xl" style="font-variation-settings: 'FILL' 1;">cloud_upload</span>
            </div>
            <div class="space-y-3">
                <h3 class="text-2xl font-bold font-headline text-white tracking-tight">Drop investment documents here</h3>
                <p class="text-slate-400 font-medium">Maximum file size: <span class="text-white">50MB</span> &bull; Supports <span class="text-white">PDF</span></p>
            </div>
            <div class="px-8 py-3 bg-white/5 rounded-2xl border border-white/10 group-hover:border-primary/30 transition-all font-semibold text-slate-300">
                Browse Files
            </div>
        </div>
        """), unsafe_allow_html=True)

        uploaded_file = st.file_uploader("Upload", type="pdf", label_visibility="collapsed")

    with r_cols[1]:
        status_text = "Ready for Ingestion" if uploaded_file else "Awaiting Selection"
        file_name  = uploaded_file.name if uploaded_file else "-"
        file_size  = (uploaded_file.size / 1024 / 1024) if uploaded_file else 0

        # FIX #2: Glass-card is fully self-contained in a single st.markdown call.
        # The Streamlit button is rendered outside the card, avoiding the unclosed-div
        # issue caused by mixing widget calls with custom HTML markup.
        st.markdown(textwrap.dedent(f"""
        <div class="glass-card p-8 border-l-4 border-l-primary/40">
            <div class="space-y-8">
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
                            <p class="text-sm font-semibold text-slate-300">{file_size:.2f} MB &bull; PDF</p>
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
            <div class="mt-8 pt-6 border-t border-white/5 flex items-center gap-3 opacity-60">
                <span class="material-symbols-outlined text-sm text-slate-400">info</span>
                <p class="text-[10px] text-slate-400 italic">Vectorization uses HuggingFace all-MiniLM-L6-v2</p>
            </div>
        </div>
        """), unsafe_allow_html=True)

        # FIX #1: Removed the duplicate info block that appeared here previously.
        # The button now renders cleanly beneath the closed glass-card div.
        if uploaded_file:
            if st.button("Begin Neural Indexing", key="btn_p", type="primary", use_container_width=True):
                with st.spinner("Analyzing Document Topology..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name
                    try:
                        loader = PyPDFLoader(tmp_path)
                        docs   = loader.load()
                        splits = RecursiveCharacterTextSplitter(
                            chunk_size=1000, chunk_overlap=200
                        ).split_documents(docs)
                        Chroma.from_documents(
                            documents=splits,
                            embedding=embeddings,
                            persist_directory=DB_DIR
                        )
                        st.session_state.process_status = "STABLE"
                        st.success("Knowledge Ingested Successfully")
                        st.rerun()
                    finally:
                        os.unlink(tmp_path)

    st.markdown("</div>", unsafe_allow_html=True)

# ==============================================================================
# TAB: SYSTEM INTEGRITY (Verify DB)
# ==============================================================================
elif st.session_state.active_tab == "Verify DB":
    st.markdown("""
    <div class="space-y-16 mt-10">
        <div class="flex items-end justify-between border-b border-white/5 pb-10">
            <div class="space-y-5">
                <div class="flex items-center gap-3">
                    <div class="flex items-center gap-2 px-3 py-1 bg-primary/10 border border-primary/20 rounded-full">
                        <span class="w-1.5 h-1.5 rounded-full bg-primary animate-pulse"></span>
                        <span class="text-primary text-[9px] font-black tracking-[0.25em] uppercase">Database Integrity</span>
                    </div>
                </div>
                <h1 class="text-6xl font-black font-headline tracking-tighter text-white leading-none">System Analytics</h1>
                <p class="text-slate-400 text-lg max-w-2xl font-medium leading-relaxed">Direct inspection of the latent document embeddings and partitioned knowledge chunks.</p>
            </div>
            <div class="flex gap-3">
                <div class="px-4 py-2 bg-slate-950/50 rounded-xl border border-white/10 text-[9px] font-black text-slate-500 uppercase tracking-widest leading-none">HuggingFace Optimized</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    if os.path.exists(DB_DIR):
        try:
            # FIX #6: Use LangChain's Chroma wrapper to access the collection so the
            # collection name always matches what the ingestion pipeline creates.
            # Previously, raw chromadb.PersistentClient + list_collections() could
            # silently inspect the wrong collection or fail across chromadb versions.
            vs         = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
            collection = vs._collection
            data       = collection.get(include=["documents", "embeddings"], limit=3)

            v_cols = st.columns(2, gap="large")

            with v_cols[0]:
                chunks_html = ""
                for i, doc in enumerate(data["documents"]):
                    chunks_html += f"""
<div class="p-5 bg-slate-900/50 rounded-2xl border border-white/5 space-y-3 group hover:border-primary/30 transition-all">
    <div class="flex items-center gap-2">
        <span class="w-2 h-2 rounded-full bg-primary opacity-40 group-hover:opacity-100 transition-opacity"></span>
        <span class="text-[10px] font-black text-slate-500 uppercase tracking-widest">Chunk {i + 1}</span>
    </div>
    <p class="text-sm text-slate-400 italic leading-relaxed">"{doc[:240]}..."</p>
</div>
"""
                with st.expander("🔍 View Knowledge Segments", expanded=False):
                    st.markdown(f"""
<div class="glass-card overflow-hidden">
    <div class="px-8 py-5 border-b border-white/5 flex justify-between items-center bg-white/5">
        <span class="font-headline font-bold text-lg text-white flex items-center gap-3">
            <span class="material-symbols-outlined text-primary">segment</span>
            Knowledge Segments
        </span>
        <span class="px-3 py-1 bg-primary/10 text-primary text-[10px] font-bold rounded-lg uppercase tracking-widest">{len(data['documents'])} Nodes</span>
    </div>
    <div class="p-8 space-y-6">{chunks_html}</div>
</div>
""", unsafe_allow_html=True)

            with v_cols[1]:
                embeddings_html = ""
                for i, emb in enumerate(data["embeddings"]):
                    embeddings_html += f"""
<div class="p-5 bg-slate-950 rounded-2xl border border-white/5 space-y-3 group hover:border-primary/30 transition-all">
    <div class="flex items-center gap-2">
        <span class="w-2 h-2 rounded-full bg-primary-400 opacity-40 group-hover:opacity-100 transition-opacity"></span>
        <span class="text-[10px] font-black text-slate-500 uppercase tracking-widest">Embedding {i + 1}</span>
    </div>
    <p class="text-[9px] font-mono text-primary/60 break-all leading-tight">[{", ".join([f"{x:.3f}" for x in emb[:15]])}...]</p>
</div>
"""
                with st.expander("📊 View Latent Embeddings", expanded=False):
                    st.markdown(f"""
<div class="glass-card overflow-hidden">
    <div class="px-8 py-5 border-b border-white/5 flex justify-between items-center bg-white/5">
        <span class="font-headline font-bold text-lg text-white flex items-center gap-3">
            <span class="material-symbols-outlined text-primary">analytics</span>
            Vector Embeddings
        </span>
        <span class="px-3 py-1 bg-primary/10 text-primary text-[10px] font-bold rounded-lg uppercase tracking-widest">d=384</span>
    </div>
    <div class="p-8 space-y-6">{embeddings_html}</div>
</div>
""", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Integrity check failed: {e}")
    else:
        st.info("Intelligence core is offline. Please ingest documents to initialize the vector database.")

    st.markdown("</div>", unsafe_allow_html=True)

# ==============================================================================
# TAB: DASHBOARD (Intelligent Query)
# ==============================================================================
elif st.session_state.active_tab == "Dashboard":
    st.markdown("""
    <div class="space-y-16 mt-10">
        <div class="flex items-end justify-between border-b border-white/5 pb-10">
            <div class="space-y-5">
                <div class="flex items-center gap-3">
                    <div class="flex items-center gap-2 px-3 py-1 bg-emerald-500/10 border border-emerald-500/20 rounded-full text-emerald-500">
                        <span class="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse"></span>
                        <span class="text-[9px] font-black tracking-[0.25em] uppercase">Cognitive Layer</span>
                    </div>
                </div>
                <h1 class="text-6xl font-black font-headline tracking-tighter text-white leading-none">Intelligent Query</h1>
                <p class="text-slate-400 text-lg max-w-2xl font-medium leading-relaxed">Probe the latent space of your ingested documents with precise, natural language inquiries.</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown(textwrap.dedent("""
    <div class="relative group mt-8">
        <div class="absolute inset-0 bg-primary/20 blur-2xl rounded-full opacity-0 group-focus-within:opacity-100 transition-opacity duration-500"></div>
    </div>
    """), unsafe_allow_html=True)

    q_in = st.text_input(
        "Ask...",
        placeholder="What are the key risk factors mentioned in the portfolio report?",
        label_visibility="collapsed"
    )

    q_cols = st.columns([1, 4, 1])
    with q_cols[1]:
        if st.button("Execute Neural Search", key="btn_q", type="primary", use_container_width=True):
            if q_in and os.path.exists(DB_DIR):
                with st.spinner("Synthesizing Insights..."):
                    vs    = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
                    chain = create_retrieval_chain(
                        vs.as_retriever(),
                        create_stuff_documents_chain(
                            get_llm(),
                            ChatPromptTemplate.from_messages([
                                ("system", """You are a precision financial analyst. Using the provided context, generate a structured Synthesis Report.

Format your response exactly as follows:
### Summary of the Provided Context
[Summarize the specific text provided in the context nodes]

### Summary of the Source Material
[If you can identify the book or source, e.g., Benjamin Graham's 'The Intelligent Investor', provide a high-level summary of its core philosophy. If unknown, skip this section.]

Use bold text for key terms and maintain a professional, insightful tone.

Context:
{context}"""),
                                ("human", "{input}"),
                            ])
                        )
                    )
                    try:
                        st.session_state.ans = chain.invoke({"input": q_in})
                    except Exception as e:
                        st.error(f"⚠️ Model invocation failed: {e}")
                        st.info("💡 This usually means the API key is invalid or the model is unavailable. Check your GOOGLE_API_KEY in Streamlit Cloud Secrets.")
            elif not os.path.exists(DB_DIR):
                st.error("No knowledge base found. Please ingest documents first.")

    if "ans" in st.session_state:
        res = st.session_state.ans

        sources_html = ""
        for i, doc in enumerate(res["context"][:2]):
            sources_html += f"""
<div class="glass-card p-6 border-l-2 border-l-primary/30 hover:border-l-primary transition-all">
    <div class="flex items-center gap-3 mb-3">
        <span class="text-[10px] font-black text-primary uppercase tracking-[0.2em]">Context node {i + 1}</span>
        <div class="h-px flex-grow bg-white/5"></div>
    </div>
    <p class="text-sm italic text-slate-400 leading-relaxed font-medium">"{doc.page_content[:320]}..."</p>
</div>
"""

        st.markdown(textwrap.dedent(f"""
        <div class="glass-card p-8 lg:p-12 space-y-12 mt-8 relative overflow-hidden">
            <div class="absolute top-0 right-0 w-64 h-64 bg-primary/5 blur-[100px] -z-10"></div>

            <div class="flex flex-col md:flex-row md:items-center justify-between gap-6">
                <div class="flex items-center gap-4">
                    <div class="w-14 h-14 bg-primary rounded-2xl flex items-center justify-center text-slate-950 shadow-xl shadow-primary/20 rotate-3">
                        <span class="material-symbols-outlined text-2xl" style="font-variation-settings: 'FILL' 1;">insights</span>
                    </div>
                    <div>
                        <h3 class="font-headline text-2xl font-black text-white tracking-tight">Synthesis Report</h3>
                        <p class="text-slate-400 text-xs font-medium">Verified Neural Integration Analysis</p>
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
        """), unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

