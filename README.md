# Stacks — AI Document Assistant

An AI-powered chatbot that answers questions based on your own documents, instead of just guessing from general knowledge. Built with Python, Flask, and the Groq LLM API.

## What it does

- Reads `.txt` and `.pdf` files placed in the `knowledge_base/` folder
- Splits documents into chunks and searches them using TF-IDF similarity
- Sends the most relevant chunks to an LLM (via Groq) along with the user's question
- Returns an answer grounded in the actual document content
- Marks responses that came from the knowledge base with a "from your documents" tag
- Custom-built chat interface (no template UI kits used)

## Tech stack

- **Backend:** Python, Flask
- **AI model:** Groq API (`openai/gpt-oss-120b`)
- **Retrieval:** scikit-learn (TF-IDF + cosine similarity)
- **PDF parsing:** pypdf
- **Frontend:** HTML, CSS, vanilla JavaScript

## Setup

1. Clone or download this project.

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Get a free API key from [console.groq.com](https://console.groq.com) (API Keys section).

4. Open `app.py` and paste your key into:
   ```python
   GROQ_API_KEY = "your-key-here"
   ```

5. Add your own documents (`.txt` or `.pdf`) into the `knowledge_base/` folder. A sample FAQ file is included to test with.

6. Run the app:
   ```
   python app.py
   ```

7. Open your browser to `http://127.0.0.1:5000`

## Project structure

```
project/
├── app.py                      # Flask backend + RAG logic
├── requirements.txt
├── knowledge_base/
│   └── company_faq.txt         # sample document (replace with your own)
└── templates/
    └── index.html              # chat interface
```

## Notes

- Conversation history resets when the server restarts (in-memory only, no database).
- Currently supports `.txt` and `.pdf` files. DOCX support planned.
- This is a demo/portfolio project — for production use, the API key should be loaded from an environment variable rather than hardcoded.

## Author

Built by Sara — AI Engineering student, exploring practical applications of LLMs for document-based Q&A.
