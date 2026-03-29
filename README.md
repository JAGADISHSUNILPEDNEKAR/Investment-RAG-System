# Investment RAG System 📈💸

A functional Retrieval-Augmented Generation (RAG) system built with Streamlit for analyzing stock market and investment textbooks. This project demonstrates an end-to-end pipeline from PDF ingestion to semantic retrieval and response generation.

## 🚀 Features
- **PDF Ingestion**: Automatically chunks and processes PDF textbooks.
- **Vector Database**: Uses **ChromaDB** for local vector storage.
- **Free Embeddings**: Uses **HuggingFace** (`all-MiniLM-L6-v2`) for local embedding generation.
- **Powerful LLM**: Integrated with **Google Gemini 1.5 Flash** for accurate answers.
- **Modern UI**: Professional Dark Mode interface with Material Icons.
- **Assignment Ready**: Specialized tabs for system initialization, backend verification, and live querying.

## 🛠️ Tech Stack
- **Frontend**: Streamlit
- **RAG Framework**: LangChain
- **Vector DB**: ChromaDB
- **Embeddings**: HuggingFace (Sentence Transformers)
- **LLM**: Google Gemini API

## 📋 Prerequisites
- Python 3.9+
- A Google Gemini API Key (get one for free at [Google AI Studio](https://aistudio.google.com/app/apikey))

## ⚙️ Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd RAG
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate  # On Windows
   source venv/bin/activate  # On Mac/Linux
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables**:
   Create a `.env` file in the root directory and add your Google API key:
   ```env
   GOOGLE_API_KEY=your_api_key_here
   ```

5. **Run the application**:
   ```bash
   streamlit run app.py
   ```

## 📚 How it Works
1. **Initialize**: Upload your investment PDF in the "Initialization" tab.
2. **Verify**: Check the "Backend Verification" tab to see how text is converted into chunks and vectors.
3. **Query**: Ask complex investment questions in the "Live Querying" tab to get context-aware answers.

## 🔍 Diagnostics & Testing
A few diagnostic scripts are available in the `tests/` directory to verify system connectivity and model availability:

- `tests/test_models.py`: Lists all models available to your Google API key.
- `tests/test_langchain.py`: Verifies LangChain's integration with Gemini.
- `tests/test_quota.py`: Tests multiple models to find the first one that works within your quota.

To run a test, use:
```bash
python tests/<test_script_name>.py
```

## 👤 Author
- **Name**: Jagadish Sunil Pednekar
- **Instructor**: Achint Setia
- **Topic**: Stock Market & Investment Analysis
