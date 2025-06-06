.PHONY: setup data clean lint test docs

setup:
	uv pip install -r requirements.txt # Use uv for installation

data:
	python src/data/preprocessing.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .ipynb_checkpoints -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

lint:
	black src
	isort src

test:
	pytest tests/

docs:
	cd docs && mkdocs build
