import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from ingest import get_vector_store

load_dotenv()

SYSTEM_PROMPT = """You are a precise financial analyst assistant.
Answer the user's question using ONLY the provided document context below.

Rules:
1. Be concise, objective, and accurate.
2. Base your response strictly on facts explicitly stated in the context.
3. If the provided context does NOT contain enough information to answer the question, state exactly: "The requested information is not available in the uploaded documents." Do not guess or invent any facts.

Context:
{context}
"""

def get_llm_model(provider: str = "openai"):
    """Instantiates the selected LLM."""
    if provider == "gemini":
        return ChatGoogleGenerativeAI(
            model="gemini-3.5-flash",  # Active Gemini Flash LLM model
            temperature=0.1,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
    else:
        return ChatOpenAI(
            model="gpt-4o",
            temperature=0.1,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )

def answer_financial_query(question: str, top_k: int = 4, provider: str = "openai"):
    """Retrieves context chunks and generates an answer."""
    vector_store = get_vector_store(provider)
    retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
    
    docs = retriever.invoke(question)
    
    if not docs:
        return "The requested information is not available in the uploaded documents.", []

    context_text = "\n\n---\n\n".join([doc.page_content for doc in docs])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}")
    ])
    
    llm = get_llm_model(provider)
    chain = prompt | llm
    
    response = chain.invoke({"context": context_text, "question": question})
    
    sources = []
    for doc in docs:
        sources.append({
            "file_name": doc.metadata.get("file_name", "Unknown File"),
            "page": doc.metadata.get("page_label", "Unknown Page"),
            "content_preview": doc.page_content[:200].replace("\n", " ") + "..."
        })
        
    return response.content, sources
