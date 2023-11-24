# Part 1: Command line chat

First, we'll be building a command line tool which takes a filename and a question as argument, indexes the file into ChromaDB, builds the prompts, sends them to an LLM and streams the result into the screen.

The full example is in `cmd_example.py`, and the below box shows how to use it. The `luoto.pdf` used as an example is a collection of text from Luoto Company website and a few articles, translated into English.

```
export OPENAI_API_KEY=<your OpenAI API Key here>

python cmd_example.py --file=luoto.pdf --question="what is luoto?"

python cmd_example.py --file=luoto.pdf --question="what is an ecosystem?"

```

## Let's build the chat

First, imports:

```python
import argparse
import asyncio
import hashlib
import os

from typing import List
import openai

import chromadb
from chromadb import Collection
from chromadb.config import Settings as ChromaDbSettings

from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader

# https://github.com/huggingface/transformers/issues/5486
# (^ to avoid a warning message, see above)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
```

### System prompt and OpenAI messages

Set up a simple system prompt template:

```python
SYSTEM_PROMPT_TPL = """
Use the following pieces of information to answer the question of the user.

<context>
{context}
</context>
"""
```

And a helper function which takes the user's question, a list of strings as context, and returns an array of messages compatible with the chat message format of OpenAI Chat Completion API.

```python
# Build the messages to be sent to OpenAI chat completion API
# System prompt + user message
def build_messages(user_question: str, context: List[str]):
    context_str = '\n\n- '.join(context)
    system_prompt = SYSTEM_PROMPT_TPL.format(context=context_str)

    print('')
    print('system prompt for LLM:')
    print('------')
    print(system_prompt)
    print('------')
    print('')

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_question}
    ]

```

### Data processing functions

Next up, a few helper functions for processing a PDF into smaller chunks to be inserted into the vector database. For the index/collection name, we hash the filename:

```python
# Each file gets its own ChromaDB index, using hash as the index name
def get_collection_name(fname: str):
    sha = hashlib.sha224()
    sha.update(fname.encode())
    return sha.hexdigest()
```

When we have text data, it is split into chunks that will be converted to vector embeddings when they are inserted into the database.

```python
# Takes a Langchain Document and splits the contents into text chunks
# of size <chunk_size> characters with <overlap> characters
def split_to_chunks(document: Document, chunk_size: int, overlap: int):
    txt = document.page_content

    # remove line changes and extra whitespace
    txt = txt.replace('\n', ' ')
    txt = ' '.join(txt.split())
    txt_len = len(txt)

    chunks = []
    for start in range(0, txt_len, (chunk_size-overlap)):
        chunks.append(Document(
            page_content=txt[start:start+chunk_size],
            metadata={
                **document.metadata,
                'idx': len(chunks)
            }
        ))

    return chunks
```

Then the function which takes a PDF filename and returns an array of text chunks:

```python
# Take a PDF filename, split it into pages and chunks, return the chunks
def parse_pdf(filename):
    loader = PyPDFLoader(filename)
    pages = loader.load_and_split()

    chunks = []
    for page in pages:
        chunks.extend(split_to_chunks(page, 500, 50))

    return chunks
```

### Building the index

Next we need a higher-level function which uses the above ones, takes a filename and returns a ChromaDB collection:

```python
# Get or create ChromaDB index for a given file
def build_index(filename: str, override_collection_name="") -> Collection:
    chroma = chromadb.PersistentClient(
        './chroma_data',
        ChromaDbSettings(
            anonymized_telemetry=False,
        )
    )

    collection_name = get_collection_name(
        override_collection_name
        if override_collection_name is not None
        else filename
    )
    collection = None
    try:
        collection = chroma.get_collection(collection_name)
        print('Found exiting collection')
    except ValueError:
        print(f'Will create a new collection for {filename}')

    if not collection:
        collection = chroma.create_collection(collection_name)
        chunks = parse_pdf(filename)
        for chunk in chunks:
            collection.add(
                documents=chunk.page_content,
                metadatas=chunk.metadata,
                ids=f"{chunk.metadata['page']}_{chunk.metadata['idx']}",
            )

    print(f'index ready for {filename}')
    return collection
```

### The main function + argument parsing

Here's the main function:

```python
async def main(filename: str, user_question: str):
    collection = build_index(filename)

    # Search for data given the user's question
    search_result = collection.query(
        query_texts=[user_question],
        n_results=5
    )
    context = ['\n'.join(doclist) for doclist in search_result['documents']]

    # Make a request to OpenAI chat completions API
    stream = openai.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=build_messages(user_question, context),
        temperature=0.25,
        top_p=0.2,
        max_tokens=512,
        stream=True
    )

    # Stream the LLM's output to screen, token by token
    for txt in stream:
        content = txt.choices[0].delta.content or ""
        print(
            content,
            end="",
            flush=True
        )

```

And finally, parse the arguments and run the main function:

```python
if __name__ == '__main__':
    parser = argparse.ArgumentParser(exit_on_error=True)
    parser.add_argument('--file', type=str, required=True)
    parser.add_argument('--question', type=str, required=True)
    args = parser.parse_args()

    asyncio.run(main(args.file, args.question))
```

Done! You should now have a command-line app that takes a file, question and streams the LLM's output to the screen.

## Part 2: Adding a UI

Next we add a user interface with file upload support, for that head into [PART2_UI.md](PART2_UI.md)
