import os
import re
from langchain_pinecone import PineconeVectorStore
from src.helper import get_embeddings
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from src.helper import get_embeddings
from src.lang_utils import detect_user_language, translate_text

# ---------------- FLASK APP ----------------

app = Flask(__name__)

# ---------------- ENVIRONMENT ----------------
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")

if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY is required. Set it in a .env file or environment.")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
if PINECONE_ENVIRONMENT:
    os.environ["PINECONE_ENVIRONMENT"] = PINECONE_ENVIRONMENT

embeddings = get_embeddings()

index_name = "medical-chatbot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# ---------------- HELPER ----------------

def process_document_content(content):

    processed = re.sub(
        r'GEM - \d+ to \d+ - [A-Z] \d{2}/\d{2}/\d{2} \d{1,2}:\d{2} [AP]M Page \d+',
        '',
        content
    )

    processed = re.sub(
        r'GALE ENCYCLOPEDIA OF MEDICINE \d+',
        '',
        processed
    )

    return processed.strip()


# ---------------- HOME ----------------

@app.route("/")
def index():

    return render_template("chat.html")


# ---------------- TEST TRANSLATION ----------------

@app.route("/test-translate")
def test_translate():

    text = request.args.get("text", "Hello")

    target_lang = request.args.get("lang", "hi")

    try:

        translated = translate_text(text, target_lang=target_lang)

        return jsonify({
            "original": text,
            "translated": translated,
            "language": target_lang,
            "status": "success"
        })

    except Exception as e:

        return jsonify({
            "status": "failed",
            "error": str(e)
        })


# ---------------- CHAT ----------------

@app.route("/get", methods=["POST"])
def chat():

    msg = request.form["msg"]

    selected_lang = request.form.get("lang", "en")

    print("User Input :", msg)

    print("Language :", selected_lang)

    greetings = ["hi", "hello", "hey", "hii"]

    if msg.lower().strip() in greetings:

        reply = "Hello! How can I assist you today?"

        if selected_lang != "en":
            reply = translate_text(reply, target_lang=selected_lang)

        return jsonify(reply)

    # Detect language

    user_lang = detect_user_language(msg, selected_lang)

    msg_en = msg

    # Translate to English

    if user_lang != "en":

        try:

            msg_en = translate_text(msg, target_lang='en')
            print("Translated Input :", msg_en)

        except Exception as e:

            print("Translation Error :", e)
            msg_en = msg

    try:

        docs = retriever.invoke(msg_en)

        processed_results = list(dict.fromkeys(

            process_document_content(
                doc.page_content
            )

            for doc in docs

        ))

        combined_response = "\n\n".join(processed_results)

        print("Response Length :", len(combined_response))

        # Translate output

        if selected_lang != "en" and combined_response:

            try:
                if len(combined_response) > 500:
                    sentences = combined_response.split(". ")
                    translated_parts = []
                    for sentence in sentences:
                        if sentence.strip():
                            translated_parts.append(
                                translate_text(sentence, target_lang=selected_lang)
                            )
                    combined_response = ". ".join(translated_parts)
                else:
                    combined_response = translate_text(combined_response, target_lang=selected_lang)
            except Exception as e:
                print("Response Translation Error :", e)

        return jsonify(combined_response)

    except Exception as e:

        print("Retriever Error :", e)
        error_message = str(e)
        error_code = None
        error_type = None

        # Parse structured error data when available
        if hasattr(e, 'args') and e.args:
            for arg in e.args:
                if isinstance(arg, dict):
                    error_code = arg.get('code') or arg.get('error', {}).get('code')
                    error_type = arg.get('type') or arg.get('error', {}).get('type')
                    break

        if error_code:
            print("Retriever Error code:", error_code)
        if error_type:
            print("Retriever Error type:", error_type)

        if any(term in error_message.lower() for term in ["quota", "429", "insufficient_quota", "rate limit"]) or error_code == 'insufficient_quota' or error_type == 'insufficient_quota':
            return jsonify(
                "Error: Retrieval failed due to quota or service limits. Please check your Pinecone indexing and retry."
            )

        return jsonify(
            "Error: Could not retrieve documents. Please try again later."
        )


# ---------------- RUN ----------------

if __name__ == "__main__":

    port = int(

        os.environ.get("PORT", 5000)

    )

    app.run(

        host="0.0.0.0",

        port=port,

        debug=True

    )