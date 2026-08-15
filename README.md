# 📈 Multi-Provider Financial RAG Desk

A Retrieval-Augmented Generation (RAG) application that allows users to upload quarterly financial reports (PDFs), index them into ChromaDB, and ask natural-language questions. The application provides answers with source citations including file names and page numbers.

---

## Features

* Upload one or more quarterly financial report PDFs
* Automatic PDF parsing and text extraction
* Chunking using Recursive Character Text Splitter
* OpenAI (`text-embedding-3-small`) or Google Gemini embeddings
* Persistent ChromaDB vector storage
* GPT-4o / Gemini powered question answering
* Source citations with file name and page number
* Honest refusal when information is unavailable
* Database persistence after restart

---

##  Project Structure

```text
finance-rag/
├── app.py                  # Streamlit UI
├── ingest.py              # PDF loading, chunking, embeddings
├── rag.py                 # Retrieval + LLM answering
├── data/                  # Quarterly PDFs
├── chroma_db_openai/      # OpenAI vector database
├── chroma_db_gemini/      # Gemini vector database
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Company & Data Used

**Company:** Infosys Ltd.

Quarterly Reports Used:

* Q1 FY26
* Q2 FY26
* Q3 FY26
* Q4 FY26

Store PDFs inside:

```text
data/
```

---

## Technology Stack

| Component             | Technology                     |
| --------------------- | ------------------------------ |
| Language              | Python 3.10+                   |
| UI                    | Streamlit                      |
| PDF Loader            | PyPDFLoader                    |
| Chunking              | RecursiveCharacterTextSplitter |
| Embeddings            | OpenAI text-embedding-3-small  |
| Vector DB             | ChromaDB                       |
| LLM                   | GPT-4o / Gemini                |
| Environment Variables | python-dotenv                  |

---

## Chunking Configuration

```python
chunk_size = 1000
chunk_overlap = 150
```

### Reason

A chunk size of 1000 characters keeps financial tables and management commentary together while maintaining good retrieval accuracy.

---

##  Environment Variables

Create a `.env` file:

```env
OPENAI_API_KEY=your_openai_api_key
GOOGLE_API_KEY=your_google_api_key
MODEL_PROVIDER=openai
```

Never upload `.env` to GitHub.

---

## .gitignore

```gitignore
.env
__pycache__/
*.pyc
chroma_db_openai/
chroma_db_gemini/
```

---

##  Installation

```bash
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

---

## ▶️ Run Application

```bash
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

---

## Workflow

1. Upload quarterly PDFs
2. Click **Index Documents**
3. Chunks are embedded and stored in ChromaDB
4. Ask financial questions
5. GPT-4o answers using retrieved chunks
6. Sources with page numbers are displayed

---

##  Sample Questions

1. What was total revenue in the latest quarter?
2. Compare net profit across quarters.
3. What was operating margin?
4. What did management say about demand outlook?
5. Which geography grew fastest?
6. Was a dividend declared?
7. What risks were mentioned?
8. Give a three-line summary.
9. Compare revenue YoY.
10. Trap question: "What was CEO shareholding in 2015?"

---

##  Honest Refusal

If information is not found:

> "The requested information is not available in the uploaded documents."

---

##  Screenshots

Add screenshots here:

<img width="797" height="136" alt="image" src="https://github.com/user-attachments/assets/a4c64db1-461b-4c5b-a515-b7d7d17ff11a" />

<img width="591" height="152" alt="image" src="https://github.com/user-attachments/assets/236f61c9-616b-4125-89c6-c23abfe38c0b" />

<img width="1035" height="486" alt="image" src="https://github.com/user-attachments/assets/b2f18084-9307-4efb-8c22-4075f08789e6" />


##  Demo Video

Record a 3-minute video demonstrating:

* Upload PDFs
* Indexing
* Three sample questions
* Trap question refusal

---

##  Future Improvements

* FastAPI backend
* REST API endpoints
* Share price integration using yfinance
* Docker deployment

---

## 👨‍💻 Author

Haripriya R

Financial RAG System using GPT-4o, ChromaDB, Streamlit, and LangChain.
