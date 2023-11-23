# A guide for building a basic chatbot

This guide shows one example of how to make a chatbot which can answer questions about PDF files.

# Building blocks

### LLM

Large Language Model. For this guide, we'll use GPT-3.5 via [OpenAI APIs](https://platform.openai.com/docs/api-reference).

In addition to GPTs, there are many other models, such as [Bard](https://bard.google.com), [Claude](https://www.anthropic.com/product), [Llama](https://ai.meta.com/llama/), [Falcon](https://falconllm.tii.ae/), [Cohere](https://cohere.com/) and [Mistral](https://mistral.ai/news/announcing-mistral-7b/).

### Vector database

A database that stores vectors. Data is converted into a vector space, where similar items get clustered closer to each other. When searching for data, the search term is converted to the same vector space, and results are then fetched using a similarity metric such as cosine similarity.

We'll use [ChromaDB](https://docs.trychroma.com/), which is an open source vector database. It uses SQLite as the default backend, which is nice and easy for simple prototyping.

Here's a few other vector databases:

- [Pinecone](https://www.pinecone.io/). Free account allows one vector index
- [OpenSearch](https://opensearch.org/platform/search/vector-database.html)
- [pgvector](https://github.com/pgvector/pgvector), a vector db extension for Postgres
- [Milvus](https://milvus.io/)
- [Qdrant](https://qdrant.tech/)
- [Marqo](https://www.marqo.ai/)

### Embedding model

Embedding model is a neural network whose job is to get data as input and output numerical vectors.

ChromaDB ships with a default embedding model, let's use that for simplicity. Basically one could use any embedding model to vectorize and then search for data, as long as you use the same model for data and the search query.

More about embeddings:

- https://huggingface.co/blog/getting-started-with-embeddings
- https://www.cloudflare.com/learning/ai/what-are-embeddings/
- https://docs.trychroma.com/embeddings
- https://platform.openai.com/docs/guides/embeddings

## Data processing

Preparing data to be stored in a vector database is a large topic and its complexity depends on how complex the data is (e.g. a simple text file vs images containing text on a PDF).

The basic idea is to chop data into short pieces that are then converted into vectors. The data can be anything, such a website, txt file, csv, pdf, transcribed audio, output from an LLM, and so on.

For this guide we'll go with the document loader api of Langchain and [PyPdfLoader](https://python.langchain.com/docs/modules/data_connection/document_loaders/pdf), which reads text from PDFs.

More about data processing:

- https://www.pinecone.io/learn/chunking-strategies/
- For a more complex set of data processing tools, there are libraries like [Unstructured](https://unstructured.io/), which also has a [Langchain integration](https://python.langchain.com/docs/integrations/providers/unstructured).

## Chat user interface

A chat UI can naturally be built with any tech, but instead of focusing on React hooks or Vue templates, we'll take an off-the-shelf solution and use a library called [Chainlit](https://docs.chainlit.io/get-started/overview).

Another library for LLM/data focused UI creation is [Streamlit](https://streamlit.io/).

## Middleware / LLM orchestration

LLM orhcestration libraries wrap the above concepts (data processing, vector databases, LLMs) into a package that's easier to use and build pipelines where one can easily switch the different components such as LLMs and vector databases. They also implement more complex use cases such as different kind of agents.

While an orchestration library is very useful in practice, it's not mandatory. For the sake of learning, we'll go without a library in this guide.

Here's a few libraries for LLM orchestration:

- [Langchain](https://python.langchain.com/docs/get_started/introduction)
- [Semantic Kernel](https://github.com/microsoft/semantic-kernel)
- [LlamaIndex](https://www.llamaindex.ai/)

# Let's begin

This guide was made with Python 3.11 and a Mac. In addition to Python and a shell, you'll be needing an OpenAI API key.

## Setup

First, create a Python virtual env and activate it:

```
python -m venv .venv
source .venv/bin/activate

(to get out of the env, call deactivate)
```

Then install the dependencies:

```
pip install -r requirements.txt

(requirements.txt is basically the contents from this)
pip install chromadb openai langchain pypdf chainlit
```

## Part 1: Building a command-line version

Head on to [PART1_CMDLINE.md](PART1_CMDLINE.md) for building the first part.
