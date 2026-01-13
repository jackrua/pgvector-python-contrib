.PHONY: lint build publish clean docs docs-serve

lint:
	pycodestyle . --ignore=E501

build:
	python3 -m build

publish: clean build
	twine upload dist/*

clean:
	rm -rf .pytest_cache dist pgvector.egg-info

docs:
	mkdocs build

docs-serve:
	mkdocs serve
