.PHONY: install run test check assistant-setup assistant-index assistant-run

install:
	python3 -m pip install -r requirements.txt

run:
	python3 -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 4173

assistant-setup:
	ollama pull qwen3:8b
	ollama pull qwen3-embedding:0.6b

assistant-index:
	SUNFINDER_ASSISTANT_ENABLED=1 python3 scripts/build_venue_index.py

assistant-run:
	SUNFINDER_ASSISTANT_ENABLED=1 python3 -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 4173

test:
	python3 -m unittest discover -s tests

check:
	python3 -m compileall -q backend
	python3 -m unittest discover -s tests
	node --check frontend/app.js
	node tests/test_client_shadows.js
