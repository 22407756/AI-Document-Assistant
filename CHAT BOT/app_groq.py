import os
import re
from flask import Flask, render_template, request, jsonify
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pypdf import PdfReader

app = Flask(__name__)

GROQ_API_KEY =os.environ.get("GROQ_API_KEY") 
client = Groq(api_key=GROQ_API_KEY)

KNOWLEDGE_BASE_FOLDER = "knowledge_base"
CHUNK_SIZE = 500

conversation_history = []


def read_txt(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def read_pdf(filepath):
    reader = PdfReader(filepath)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


def load_documents():
    all_text = ""
    if not os.path.exists(KNOWLEDGE_BASE_FOLDER):
        os.makedirs(KNOWLEDGE_BASE_FOLDER)
        return all_text

    for filename in os.listdir(KNOWLEDGE_BASE_FOLDER):
        filepath = os.path.join(KNOWLEDGE_BASE_FOLDER, filename)
        if filename.lower().endswith(".txt"):
            all_text += read_txt(filepath) + "\n"
        elif filename.lower().endswith(".pdf"):
            all_text += read_pdf(filepath) + "\n"

    return all_text


def chunk_text(text, chunk_size=CHUNK_SIZE):
    text = re.sub(r"\s+", " ", text).strip()
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end
    return [c for c in chunks if c.strip()]


print("Loading knowledge base...")
raw_text = load_documents()
chunks = chunk_text(raw_text) if raw_text else []

if chunks:
    vectorizer = TfidfVectorizer(stop_words="english")
    chunk_vectors = vectorizer.fit_transform(chunks)
    print(f"Loaded {len(chunks)} chunks from your documents.")
else:
    vectorizer = None
    chunk_vectors = None
    print("No documents found in knowledge_base/ — bot will answer from general knowledge only.")


def find_relevant_chunks(question, top_k=3):
    if not chunks or vectorizer is None:
        return []

    question_vector = vectorizer.transform([question])
    similarities = cosine_similarity(question_vector, chunk_vectors)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    relevant = [chunks[i] for i in top_indices if similarities[i] > 0.05]
    return relevant


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message", "")

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    relevant_chunks = find_relevant_chunks(user_message)

    if relevant_chunks:
        context = "\n\n---\n\n".join(relevant_chunks)
        system_prompt = (
            "You are a helpful assistant. Answer the user's question using ONLY the "
            "context below. If the answer isn't in the context, say you don't have "
            "that information in your documents.\n\n"
            f"CONTEXT:\n{context}"
        )
    else:
        system_prompt = "You are a helpful assistant."

    conversation_history.append({"role": "user", "content": user_message})
    messages = [{"role": "system", "content": system_prompt}] + conversation_history

    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            max_tokens=500,
            messages=messages
        )
        ai_reply = response.choices[0].message.content
        conversation_history.append({"role": "assistant", "content": ai_reply})

        return jsonify({"reply": ai_reply, "used_document_context": bool(relevant_chunks)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/reset", methods=["POST"])
def reset():
    conversation_history.clear()
    return jsonify({"status": "conversation reset"})


if __name__ == "__main__":
    app.run(debug=True)