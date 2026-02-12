.PHONY: install test lint format docker-build docker-run clean

install:
	pip install --upgrade pip
	pip install -r requirements.txt

test:
	pytest tests/ -v

lint:
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

format:
	black .

docker-build:
	docker build -t recsys-engine .

docker-run:
	docker run -p 8501:8501 recsys-engine

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
