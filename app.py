import asyncio
import os
import re

import googletrans
from flask import Flask, jsonify, render_template, request
from googletrans import Translator
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from src.helper import download_hugging_face_embeddings

app = Flask(__name__)

# Set Pinecone API key directly
os.environ["PINECONE_API_KEY"] = "pcsk_4RVQVe_9fvcvhbB8dFmPk7LQXeQgEXhuWsJUHNctT2H8Qn7SHWgYrKMY1VQapdFmoxX6iY"

# Load embeddings model
embeddings = download_hugging_face_embeddings()

# Connect to Pinecone index
index_name = "medical-chatbot"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

# Create retriever
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})



translator = Translator()
from src.lang_utils import detect_user_language, translate_text


def process_document_content(content):
    """Extract meaningful information from document content"""
    processed = re.sub(r'GEM - \d+ to \d+ - [A-Z] \d{2}/\d{2}/\d{2} \d{1,2}:\d{2} [AP]M Page \d+', '', content)
    processed = re.sub(r'GALE ENCYCLOPEDIA OF MEDICINE \d+', '', processed)
    return processed.strip()

@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/test-translate", methods=["GET", "POST"])
def test_translate():
    """Test endpoint to verify translation functionality"""
    text = request.args.get("text", "Hello, how are you?")
    target_lang = request.args.get("lang", "hi")
    
    try:
        translated = translator.translate(text, dest=target_lang).text
        return jsonify({
            "original": text,
            "translated": translated,
            "target_language": target_lang,
            "status": "success"
        })
    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "failed"
        })


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    selected_lang = request.form.get("lang", "en")
    print("User Input:", msg)
    print("Selected language:", selected_lang)

    user_lang = detect_user_language(msg, selected_lang)

# 🌟 Greeting Detection
    greetings = ["hi", "hello", "hey", "hii"]

    if msg.lower().strip() in greetings:
        greeting_reply = "Hello! How can I assist you today?"
        if selected_lang != "en":
            try:
                greeting_reply = translator.translate(greeting_reply, dest=selected_lang).text
                print(f"Translated greeting to {selected_lang}:", greeting_reply)
            except Exception as e:
                print(f"❌ Translation error for greeting: {e}")
        return jsonify(greeting_reply)
    
    # Translate user input to English if not already English
    msg_en = msg
    if user_lang != "en":
        try:
            msg_en = translator.translate(msg, dest="en").text
            print("Translated to English:", msg_en)
        except Exception as e:
            print(f"❌ Translation error for user input: {e}")
            msg_en = msg

    try:
        docs = retriever.invoke(msg_en)
        processed_results = list(dict.fromkeys(
            process_document_content(doc.page_content) for doc in docs
        ))
        combined_response = "\n\n".join(processed_results)
        print("Combined response length:", len(combined_response))

        # Translate response to selected language if needed
        if selected_lang != "en" and combined_response:
            try:
                # Split long text for better translation
                if len(combined_response) > 500:
                    sentences = combined_response.split('. ')
                    translated_parts = []
                    for sentence in sentences:
                        if sentence.strip():
                            try:
                                translated = translator.translate(sentence, dest=selected_lang).text
                                translated_parts.append(translated)
                            except:
                                translated_parts.append(sentence)
                    combined_response = '. '.join(translated_parts)
                else:
                    combined_response = translator.translate(combined_response, dest=selected_lang).text
                
                print(f"Translated response to {selected_lang}")
            except Exception as e:
                print(f"❌ Translation error for response: {e}")
                # Response will be returned in English if translation fails

        return jsonify(combined_response)

    except Exception as e:
        print("❌ Error occurred:", str(e))
        return jsonify("Error: Could not retrieve documents. Please try again.")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

print("googletrans version:", googletrans.__version__)

