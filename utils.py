import hashlib
import chromadb
from chromadb import Collection
from chromadb.config import Settings as ChromaDbSettings
from chromadb.errors import InvalidCollectionException

from langchain.docstore.document import Document
from langchain_community.document_loaders import PyPDFLoader

from typing import List


# Each file gets its own ChromaDB index, using hash as the index name
def get_collection_name(fname: str):
    sha = hashlib.sha224()
    sha.update(fname.encode())
    return sha.hexdigest()


# Takes a Langchain Document and splits the contents into text chunks
# of size <chunk_size> characters with <overlap> characters
def split_to_chunks(document: Document, chunk_size: int, overlap: int):
    txt = document.page_content

    # remove line changes and extra whitespace
    txt = txt.replace("\n", " ")
    txt = " ".join(txt.split())
    txt_len = len(txt)

    chunks = []
    for start in range(0, txt_len, (chunk_size - overlap)):
        chunks.append(
            Document(
                page_content=txt[start : start + chunk_size],
                metadata={**document.metadata, "idx": len(chunks)},
            )
        )

    return chunks


# Take a PDF filename, split it into pages and chunks, return the chunks
def parse_pdf(filename):
    loader = PyPDFLoader(filename)
    pages = loader.load_and_split()

    chunks = []
    for page in pages:
        chunks.extend(split_to_chunks(page, 500, 50))

    return chunks


# Get or create ChromaDB index for a given file
def build_index(filename: str, override_collection_name="") -> Collection:
    chroma = chromadb.PersistentClient(
        "./chroma_data",
        ChromaDbSettings(
            anonymized_telemetry=False,
        ),
    )

    collection_name = get_collection_name(
        override_collection_name
        if override_collection_name is not None
        else filename
    )
    collection = None
    try:
        collection = chroma.get_collection(collection_name)
        print("Found exiting collection")
    except InvalidCollectionException:
        print(f"Will create a new collection for {filename}")

    if not collection:
        collection = chroma.create_collection(collection_name)
        chunks = parse_pdf(filename)
        for chunk in chunks:
            collection.add(
                documents=chunk.page_content,
                metadatas=chunk.metadata,
                ids=f"{chunk.metadata['page']}_{chunk.metadata['idx']}",
            )

    print(f"index ready for {filename}")
    return collection


SYSTEM_PROMPT_TPL = """
Use the following pieces of information to answer the question of the user.

<context>
{context}
</context>
"""


# Build the messages to be sent to OpenAI chat completion API
# System prompt + user message
def build_messages(user_question: str, context: List[str]):
    context_str = "\n\n- ".join(context)
    system_prompt = SYSTEM_PROMPT_TPL.format(context=context_str)

    print("")
    print("system prompt for LLM:")
    print("------")
    print(system_prompt)
    print("------")
    print("")

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_question},
    ]
