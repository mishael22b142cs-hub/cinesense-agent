from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()  # .env-ൽ നിന്ന് key എടുക്കുന്നു

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.7)

response = llm.invoke("Suggest one good Malayalam thriller movie in one line.")
print(response.content)