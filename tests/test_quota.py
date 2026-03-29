import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
models = ["gemini-1.5-flash", "gemini-1.5-flash-latest", "gemini-1.5-pro", "gemini-pro", "gemini-flash-latest", "gemini-pro-latest", "gemini-1.5-flash-8b"]

for model in models:
    try:
        print(f"Testing {model}...")
        llm = ChatGoogleGenerativeAI(model=model, temperature=0)
        res = llm.invoke("Hi")
        print(f"✅ Success with {model}!")
        break
    except Exception as e:
        print(f"❌ Error with {model}: {e}")
else:
    print("No models worked.")
