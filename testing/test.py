from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
load_dotenv()

groq_api_key=os.getenv("groq_api_key")

model="llama-3.1-8b-instant"


mm=ChatGroq(model=model,api_key=groq_api_key)
response=mm.invoke("what is the capital of india ").content
print(response)
