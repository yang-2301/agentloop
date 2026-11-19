.PHONY: install test dev serve build

install:
	python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
	cd web && npm install

test:
	.venv/bin/python -m pytest tests/ -q -p no:anyio

serve:            ## backend on :8124 (mock LLM if no OPENAI_API_KEY)
	.venv/bin/uvicorn api.main:app --port 8124

dev:              ## frontend on :5175 (proxies /api to :8124)
	cd web && npm run dev

build:
	cd web && npm run build
