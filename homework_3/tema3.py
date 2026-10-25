import os
from pathlib import Path

import chromadb
import pymupdf4llm
from dotenv import load_dotenv
from openai import OpenAI, APIError
from sentence_transformers import SentenceTransformer



MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "Homework_3"
CURRENT_DIR = Path(__file__).resolve().parent
DATA_SOURCE_DIR = CURRENT_DIR / "data_sources"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

def chunk_text(text: str) -> list[str]:
    chunks = []
    start = 0

    while start < len(text):
        chunk = text[start : start + CHUNK_SIZE].strip()
        if chunk:
            chunks.append(chunk)
        start = start + CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


def index_file(collection, model, file_path):
    pages = pymupdf4llm.to_markdown(str(file_path), page_chunks=True)
    records = []

    for page in pages:
        page_number = page["metadata"]["page_number"]
        page_text = page["text"]

        chunks = chunk_text(page_text)
        chunk_number = 0

        for chunk in chunks:
            records.append((chunk, page_number, chunk_number))
            chunk_number += 1

    collection.delete(where={"source": file_path.name})
    ids = []
    documents = []
    metadatas = []

    for record in records:
        chunk = record[0]
        page_number = record[1]
        chunk_number = record[2]
        ids.append(f"{file_path.name}:p{page_number}:c{chunk_number}")
        documents.append(chunk)
        metadatas.append(
            {
                "source": file_path.name,
                "page": page_number,
                "chunk": chunk_number,
            }
        )

    collection.upsert(
        ids=ids,

        embeddings=model.encode(documents, normalize_embeddings=True).tolist(),
        documents=documents,
        metadatas=metadatas,
    )

    return documents, metadatas


def search_files(collection,model,file_name,question):
    results = collection.get(
        where={"source": file_name},
        limit=1,
    )
    if len(results["ids"]) == 0:
        return []
    else:
        query_embeddings = model.encode(question,normalize_embeddings=True).tolist()
        results = collection.query(
            query_embeddings=[query_embeddings],
            n_results=3,
            where={"source": file_name},
            include=["documents","metadatas","distances"]
        )
        return results

def answer_with_context(client: OpenAI, results, question):
    try:
        context_pieces = []
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]

        for i in range(len(documents)):
            document = documents[i]
            metadata = metadatas[i]
            context_pieces.append(
                f"{document}\nSource: {metadata['source']}, page {metadata['page']}"
            )

        context = "\n\n".join(context_pieces)
        return client.chat.completions.create(
            model=os.getenv("OPENROUTER_MODEL"),
            messages=[
                {
                    "role": "system",
                    "content":
                            "Answer only from the supplied context. "
                            "If it does not answer the question, say so. "
                            "Cite every factual claim with the provided file name and page number.",
                },
                {"role": "user",
                 "content": f"Question: {question}\n\nContext:\n{context}"
                 },
            ],
        ).choices[0].message.content
    except APIError as error:
        return f"LLM request failed: {error}"

def main():
    load_dotenv()
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )
    model = SentenceTransformer(MODEL_NAME)
    collection = chromadb.PersistentClient().get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    available_files = []
    for file_path in DATA_SOURCE_DIR.glob("*.pdf"):
        file_documents, file_metadatas = index_file(collection, model, Path(file_path))

        if file_documents:
            available_files.append(file_path.name)

    print(f"Available files:\n"+"\n".join(available_files))

    while True:
        file_name = input("you> Enter file name: ").strip()

        if file_name.lower() == "quit":
            break

        if file_name not in available_files:
            print("Unknown file. Pick one of the following:")
            print("\n".join(available_files))
            continue

        question = input(f"you> What question do you have about {file_name}?\n").strip()

        if not question:
            print(f"Question cannot be empty")
            continue
        if question.lower() == "quit":
            break

        results = search_files(collection, model, file_name, question)

        if not results or not results["documents"] or not results["documents"][0]:
            print(f"No results found for {file_name}")
            continue

        output = answer_with_context(client, results, question)
        print(f"agent> {output}")

if __name__ == "__main__":
    main()
