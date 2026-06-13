

#RAG Chatbot for Medical
### STEP 01- Create a conda environment after opening the repository

```bash
conda create -n medibot python=3.10 -y
```

```bash
conda activate medibot
```


### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```


### Create a `.env` file in the root directory and add your Pinecone key only
```ini
PINECONE_API_KEY="xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
# Optional: PINECONE_ENVIRONMENT="us-east1-gcp"
```

```bash
# run the following command to store embeddings to pinecone
python store_index.py
```

### Option 2: Embed locally, deploy retrieval only
1. Run `python store_index.py` on your laptop.
2. Generate embeddings locally with Hugging Face and upload vectors to Pinecone.
3. Deploy only `app.py` to Render.
4. Ensure Render has `PINECONE_API_KEY` in its environment.

At deployment, `app.py` uses:
```python
docsearch = PineconeVectorStore.from_existing_index(
    index_name="medical-chatbot",
    embedding=embeddings
)
```

Use the same model locally and in deployment:
```python
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```

Remove OpenAI completely from your deployed app. Only connect to Pinecone and retrieve results; do not recreate embeddings in Render.

> If you receive a `429` / `insufficient_quota` error while developing locally, it is not related to this local HF embedding setup. Remove OpenAI API keys from your deployment environment to avoid OpenAI usage entirely.

```bash
# Finally run the following command
python app.py
```


### Techstack Used:

- Python
- LangChain
- Flask
- GPT
- Pinecone

