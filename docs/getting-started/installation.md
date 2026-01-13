# Installation

## Requirements

- Python >= 3.9
- PostgreSQL with pgvector extension installed

## Install pgvector

First, install the Python package:

```sh
pip install pgvector
```

## Database Library Setup

pgvector-python supports multiple database libraries. Choose the one you're using and follow the setup instructions below.

### Django

Create a migration to enable the extension:

```python
from pgvector.django import VectorExtension

class Migration(migrations.Migration):
    operations = [
        VectorExtension()
    ]
```

Add a vector field to your model:

```python
from pgvector.django import VectorField

class Item(models.Model):
    embedding = VectorField(dimensions=3)
```

Also supports `HalfVectorField`, `BitField`, and `SparseVectorField`

### SQLAlchemy

Enable the extension:

```python
from sqlalchemy import text

session.execute(text('CREATE EXTENSION IF NOT EXISTS vector'))
```

Add a vector column:

```python
from pgvector.sqlalchemy import Vector

class Item(Base):
    embedding = mapped_column(Vector(3))
```

Also supports `HALFVEC`, `BIT`, and `SPARSEVEC`

### SQLModel

Enable the extension:

```python
from sqlalchemy import text

session.exec(text('CREATE EXTENSION IF NOT EXISTS vector'))
```

Add a vector column:

```python
from typing import Any
from sqlmodel import Field
from pgvector.sqlalchemy import Vector

class Item(SQLModel, table=True):
    embedding: Any = Field(sa_type=Vector(3))
```

Also supports `HALFVEC`, `BIT`, and `SPARSEVEC`

### Psycopg 3

Enable the extension:

```python
conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
```

Register the types with your connection:

```python
from pgvector.psycopg import register_vector

register_vector(conn)
```

For async connections, use:

```python
from pgvector.psycopg import register_vector_async

await register_vector_async(conn)
```

### Psycopg 2

Enable the extension:

```python
cur = conn.cursor()
cur.execute('CREATE EXTENSION IF NOT EXISTS vector')
```

Register the types with your connection or cursor:

```python
from pgvector.psycopg2 import register_vector

register_vector(conn)
```

### asyncpg

Enable the extension:

```python
await conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
```

Register the types with your connection:

```python
from pgvector.asyncpg import register_vector

await register_vector(conn)
```

Or your pool:

```python
import asyncpg

async def init(conn):
    await register_vector(conn)

pool = await asyncpg.create_pool(..., init=init)
```

### pg8000

Enable the extension:

```python
conn.run('CREATE EXTENSION IF NOT EXISTS vector')
```

Register the types with your connection:

```python
from pgvector.pg8000 import register_vector

register_vector(conn)
```

### Peewee

Add a vector column:

```python
from pgvector.peewee import VectorField

class Item(BaseModel):
    embedding = VectorField(dimensions=3)
```

Also supports `HalfVectorField`, `FixedBitField`, and `SparseVectorField`

## Next Steps

- Check out the [examples](../examples/openai.md) to see how to use pgvector with different services
- Learn about vector operations and indexing in the full documentation
