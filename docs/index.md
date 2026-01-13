# pgvector-python

[pgvector](https://github.com/pgvector/pgvector) support for Python

Supports [Django](https://github.com/django/django), [SQLAlchemy](https://github.com/sqlalchemy/sqlalchemy), [SQLModel](https://github.com/tiangolo/sqlmodel), [Psycopg 3](https://github.com/psycopg/psycopg), [Psycopg 2](https://github.com/psycopg/psycopg2), [asyncpg](https://github.com/MagicStack/asyncpg), [pg8000](https://github.com/tlocke/pg8000), and [Peewee](https://github.com/coleifer/peewee)

[![Build Status](https://github.com/pgvector/pgvector-python/actions/workflows/build.yml/badge.svg)](https://github.com/pgvector/pgvector-python/actions)

## Quick Start

Install the package:

```sh
pip install pgvector
```

Follow the instructions for your database library:

- [Django](getting-started/installation.md#django)
- [SQLAlchemy](getting-started/installation.md#sqlalchemy)
- [SQLModel](getting-started/installation.md#sqlmodel)
- [Psycopg 3](getting-started/installation.md#psycopg-3)
- [Psycopg 2](getting-started/installation.md#psycopg-2)
- [asyncpg](getting-started/installation.md#asyncpg)
- [pg8000](getting-started/installation.md#pg8000)
- [Peewee](getting-started/installation.md#peewee)

## Examples

Check out these examples to get started:

- [Retrieval-augmented generation](https://github.com/pgvector/pgvector-python/blob/master/examples/rag/example.py) with Ollama
- [Embeddings](examples/openai.md) with OpenAI
- [Binary embeddings](https://github.com/pgvector/pgvector-python/blob/master/examples/cohere/example.py) with Cohere
- [Sentence embeddings](https://github.com/pgvector/pgvector-python/blob/master/examples/sentence_transformers/example.py) with SentenceTransformers
- [Hybrid search](https://github.com/pgvector/pgvector-python/blob/master/examples/hybrid_search/rrf.py) with SentenceTransformers (Reciprocal Rank Fusion)
- [Hybrid search](https://github.com/pgvector/pgvector-python/blob/master/examples/hybrid_search/cross_encoder.py) with SentenceTransformers (cross-encoder)
- [Sparse search](https://github.com/pgvector/pgvector-python/blob/master/examples/sparse_search/example.py) with Transformers
- [Late interaction search](https://github.com/pgvector/pgvector-python/blob/master/examples/colbert/exact.py) with ColBERT
- [Visual document retrieval](https://github.com/pgvector/pgvector-python/blob/master/examples/colpali/exact.py) with ColPali
- [Image search](https://github.com/pgvector/pgvector-python/blob/master/examples/image_search/example.py) with PyTorch
- [Image search](https://github.com/pgvector/pgvector-python/blob/master/examples/imagehash/example.py) with perceptual hashing
- [Morgan fingerprints](https://github.com/pgvector/pgvector-python/blob/master/examples/rdkit/example.py) with RDKit
- [Topic modeling](https://github.com/pgvector/pgvector-python/blob/master/examples/gensim/example.py) with Gensim
- [Implicit feedback recommendations](https://github.com/pgvector/pgvector-python/blob/master/examples/implicit/example.py) with Implicit
- [Explicit feedback recommendations](https://github.com/pgvector/pgvector-python/blob/master/examples/surprise/example.py) with Surprise
- [Recommendations](https://github.com/pgvector/pgvector-python/blob/master/examples/lightfm/example.py) with LightFM
- [Horizontal scaling](https://github.com/pgvector/pgvector-python/blob/master/examples/citus/example.py) with Citus
- [Bulk loading](https://github.com/pgvector/pgvector-python/blob/master/examples/loading/example.py) with `COPY`

## Features

- **Multiple Database Adapters**: Works seamlessly with popular Python database libraries
- **Vector Operations**: Support for L2 distance, inner product, cosine distance, and more
- **Approximate Indexes**: HNSW and IVFFlat index support for fast similarity search
- **Vector Types**: Support for regular vectors, half-precision vectors, bit vectors, and sparse vectors
- **Batch Operations**: Efficient bulk loading and batch processing capabilities

## Contributing

Everyone is encouraged to help improve this project. Here are a few ways you can help:

- [Report bugs](https://github.com/pgvector/pgvector-python/issues)
- Fix bugs and [submit pull requests](https://github.com/pgvector/pgvector-python/pulls)
- Write, clarify, or fix documentation
- Suggest or add new features

## History

View the [changelog](https://github.com/pgvector/pgvector-python/blob/master/CHANGELOG.md)
