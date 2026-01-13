# OpenAI Embeddings Example

This example demonstrates how to use pgvector with OpenAI's embedding API to store and search text embeddings.

## Overview

This example shows how to:

- Generate embeddings using OpenAI's API
- Store embeddings in PostgreSQL with pgvector
- Perform similarity search to find related documents

## Prerequisites

- OpenAI API key
- PostgreSQL with pgvector extension installed
- Python packages: `openai`, `pgvector`, `psycopg` or another supported database adapter

## Installation

Install the required packages:

```sh
pip install pgvector openai psycopg[binary]
```

## Basic Example

Here's a simple example using Psycopg 3:

```python
import openai
import psycopg
from pgvector.psycopg import register_vector

# Set up OpenAI API
openai.api_key = 'your-api-key'

# Connect to database
conn = psycopg.connect(dbname='mydb')
register_vector(conn)

# Enable the extension
conn.execute('CREATE EXTENSION IF NOT EXISTS vector')

# Create a table
conn.execute('CREATE TABLE documents (id bigserial PRIMARY KEY, content text, embedding vector(1536))')

# Generate and store embeddings
def add_document(content):
    response = openai.embeddings.create(
        input=content,
        model="text-embedding-3-small"
    )
    embedding = response.data[0].embedding
    conn.execute('INSERT INTO documents (content, embedding) VALUES (%s, %s)', (content, embedding))

# Add some documents
add_document('The cat sits on the mat')
add_document('A dog runs in the park')
add_document('Feline animals are independent')

conn.commit()

# Search for similar documents
def search(query, limit=5):
    response = openai.embeddings.create(
        input=query,
        model="text-embedding-3-small"
    )
    embedding = response.data[0].embedding
    
    results = conn.execute(
        'SELECT content, embedding <=> %s as distance FROM documents ORDER BY distance LIMIT %s',
        (embedding, limit)
    ).fetchall()
    
    return results

# Find documents similar to a query
results = search('cat')
for content, distance in results:
    print(f'{content}: {distance}')
```

## Using with SQLAlchemy

Here's the same example using SQLAlchemy:

```python
import openai
from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import Session, DeclarativeBase, Mapped, mapped_column
from pgvector.sqlalchemy import Vector

# Set up database
engine = create_engine('postgresql://user:password@localhost/dbname')

class Base(DeclarativeBase):
    pass

class Document(Base):
    __tablename__ = 'documents'
    
    id: Mapped[int] = mapped_column(primary_key=True)
    content: Mapped[str]
    embedding: Mapped[list] = mapped_column(Vector(1536))

# Create tables
with Session(engine) as session:
    session.execute(text('CREATE EXTENSION IF NOT EXISTS vector'))
    session.commit()

Base.metadata.create_all(engine)

# Generate and store embeddings
def add_document(content):
    response = openai.embeddings.create(
        input=content,
        model="text-embedding-3-small"
    )
    embedding = response.data[0].embedding
    
    with Session(engine) as session:
        doc = Document(content=content, embedding=embedding)
        session.add(doc)
        session.commit()

# Search for similar documents
def search(query, limit=5):
    response = openai.embeddings.create(
        input=query,
        model="text-embedding-3-small"
    )
    embedding = response.data[0].embedding
    
    with Session(engine) as session:
        results = session.scalars(
            select(Document)
            .order_by(Document.embedding.l2_distance(embedding))
            .limit(limit)
        ).all()
        
        return results
```

## Performance Tips

### Add an Index

For better performance with larger datasets, add an HNSW index:

```python
conn.execute('CREATE INDEX ON documents USING hnsw (embedding vector_l2_ops)')
```

### Use Half-Precision Vectors

To save storage space, you can use half-precision vectors:

```python
# Create table with halfvec
conn.execute('CREATE TABLE documents (id bigserial PRIMARY KEY, content text, embedding halfvec(1536))')

# Index with half-precision
conn.execute('CREATE INDEX ON documents USING hnsw (embedding halfvec_l2_ops)')
```

## Complete Example

For a complete working example, see the [example.py](https://github.com/pgvector/pgvector-python/blob/master/examples/openai/example.py) file in the repository.

## Next Steps

- Learn about [hybrid search](https://github.com/pgvector/pgvector-python/blob/master/examples/hybrid_search/rrf.py) combining vector and keyword search
- Explore [RAG (Retrieval-Augmented Generation)](https://github.com/pgvector/pgvector-python/blob/master/examples/rag/example.py) patterns
- Try other embedding providers like [Cohere](https://github.com/pgvector/pgvector-python/blob/master/examples/cohere/example.py) or [SentenceTransformers](https://github.com/pgvector/pgvector-python/blob/master/examples/sentence_transformers/example.py)
