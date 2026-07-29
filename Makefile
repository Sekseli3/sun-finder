.PHONY: install run test check assistant-setup assistant-index assistant-run assistant-benchmark vllm-chat vllm-embeddings

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

assistant-benchmark:
	SUNFINDER_ASSISTANT_ENABLED=1 python3 scripts/benchmark_planner_intents.py $(BENCHMARK_ARGS)

vllm-chat:
	set -a; [ ! -f .env ] || . ./.env; set +a; vllm serve $${SUNFINDER_VLLM_CHAT_MODEL:-Qwen/Qwen3-8B} --host 127.0.0.1 --port 8000 --api-key $${SUNFINDER_VLLM_API_KEY:-sunfinder-local} --default-chat-template-kwargs '{"enable_thinking": false}'

vllm-embeddings:
	set -a; [ ! -f .env ] || . ./.env; set +a; vllm serve $${SUNFINDER_VLLM_EMBEDDING_MODEL:-Qwen/Qwen3-Embedding-0.6B} --runner pooling --host 127.0.0.1 --port 8001 --api-key $${SUNFINDER_VLLM_API_KEY:-sunfinder-local}

test:
	python3 -m unittest discover -s tests

check:
	python3 -m compileall -q backend
	python3 -m unittest discover -s tests
	node --check frontend/app.js
	node tests/test_client_shadows.js
