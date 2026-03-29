import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
try:
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    res = llm.invoke("Hi")
    print("Success with gemini-1.5-flash")
    print(res.content)
except Exception as e:
    print(f"Error with gemini-1.5-flash: {e}")

try:
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)
    res = llm.invoke("Hi")
    print("Success with gemini-pro")
    print(res.content)
except Exception as e:
    print(f"Error with gemini-pro: {e}")
