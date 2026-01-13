# Documentation

This directory contains the documentation for pgvector-python, built with [MkDocs](https://www.mkdocs.org/) and the [Material theme](https://squidfunk.github.io/mkdocs-material/).

## Building the Documentation

To build the documentation locally:

```sh
pip install mkdocs mkdocs-material
make docs
```

The built documentation will be in the `site/` directory.

## Serving the Documentation

To serve the documentation locally for development:

```sh
make docs-serve
```

This will start a development server at `http://127.0.0.1:8000/`.

## Documentation Structure

- `docs/index.md` - Home page
- `docs/getting-started/` - Getting started guides
  - `installation.md` - Installation instructions for different database adapters
- `docs/examples/` - Example usage guides
  - `openai.md` - OpenAI embeddings example

## Adding New Pages

1. Create a new Markdown file in the appropriate directory under `docs/`
2. Add the page to the navigation in `mkdocs.yml`
3. Build and test locally with `make docs-serve`

## Configuration

The documentation configuration is in `mkdocs.yml` at the root of the repository.
