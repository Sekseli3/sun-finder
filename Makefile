.PHONY: install run test check

install:
	python3 -m pip install -r requirements.txt

run:
	python3 -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 4173

test:
	python3 -m unittest discover -s tests

check:
	python3 -m compileall -q backend
	python3 -m unittest discover -s tests
	node --check frontend/app.js
	node tests/test_client_shadows.js
