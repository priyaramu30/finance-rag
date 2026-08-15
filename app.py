import os
import streamlit as st
import tempfile
from ingest import process_and_index_pdfs, get_vector_store
from rag import answer_financial_query

st.set_page_config(page_title="Multi-Provider Financial RAG Desk", layout="wide")
st.title("📈 Quarterly Financial Reports Analyst Desk")

# Sidebar - Settings & Ingestion
with st.sidebar:
    st.header("⚙️ Model Settings")
    provider_choice = st.selectbox(
        "Select AI Provider:",
        options=["openai", "gemini"],
        format_func=lambda x: "OpenAI (GPT-4o)" if x == "openai" else "Google Gemini (3.5 Flash)"
    )

    st.markdown("---")
    st.header("1. Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload Quarterly Results / Press Releases", 
        type=["pdf"], 
        accept_multiple_files=True
    )
    
    if st.button("Index Documents", type="primary"):
        if uploaded_files:
            with st.spinner(f"Processing & indexing PDFs into ChromaDB using {provider_choice.upper()}..."):
                temp_paths = []
                for uploaded_file in uploaded_files:
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                    tmp.write(uploaded_file.getvalue())
                    tmp.close()
                    temp_paths.append(tmp.name)
                        
                num_files, num_chunks = process_and_index_pdfs(temp_paths, provider=provider_choice)
                
                for p in temp_paths:
                    if os.path.exists(p):
                        os.remove(p)
                    
                st.success(f"Successfully added {num_chunks} new chunks from {num_files} file(s)!")
        else:
            st.warning("Please upload at least one PDF file first.")

    st.markdown("---")
    st.header("Database Info")
    try:
        vs = get_vector_store(provider=provider_choice)
        count = vs._collection.count()
        st.info(f"📁 Total Chunks in DB ({provider_choice.upper()}): **{count}**")
    except Exception:
        st.info("📁 Total Chunks in DB: **0**")

# Main Interface - Query Section
st.header("2. Ask Financial Questions")
query = st.text_input("Enter your question:", placeholder="e.g., What was total revenue in the most recent quarter?")

if st.button("Submit Question", type="primary"):
    if not query.strip():
        st.warning("Please enter a question first.")
    else:
        with st.spinner(f"Searching documents & querying {provider_choice.upper()}..."):
            answer, sources = answer_financial_query(query, provider=provider_choice)
            
            st.subheader(f"Answer (via {provider_choice.upper()})")
            st.markdown(f"> {answer}")
            
            st.markdown("---")
            st.subheader("Sources Cited")
            if sources:
                for idx, src in enumerate(sources, 1):
                    with st.expander(f"Source {idx}: {src['file_name']} (Page {src['page']})"):
                        st.caption(f"**Chunk Preview:** {src['content_preview']}")
            else:
                st.caption("No context sources were retrieved.")
