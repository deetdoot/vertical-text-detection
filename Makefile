install:
	pip install -r requirements.txt

run:
	uvicorn app:app --reload --port $(or $(PORT),8000)

clean-temp:
	rm -rf static/temp/*

lint:	
	flake8 .

test:
	pytest
