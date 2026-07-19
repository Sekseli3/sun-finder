.PHONY: install run check

install:
	python3 -m pip install -r requirements.txt

run:
	python3 -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 4173

check:
	python3 -m compileall -q backend
