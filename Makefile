.PHONY: help install test test-py test-course dev build run-phase1 run-phase2 run-phase3 run-phase4 run-phase5 run-phase6 run-all clean

help:
	@echo "LLM From First Principles"
	@echo "-------------------------"
	@echo "make install     : install the Python dependencies"
	@echo "make dev         : run the interactive course locally"
	@echo "make build       : type-check and build the course into course/dist"
	@echo "make test        : run everything (Python tests, course tests, type check)"
	@echo "make test-py     : run the Python tests only"
	@echo "make test-course : run the course tests only"
	@echo "make run-phase1  : foundations (maths, gradient descent, the MLP)"
	@echo "make run-phase2  : language (tokenizer, word vectors, bigram model)"
	@echo "make run-phase3  : transformers (attention, tiny GPT, KV cache)"
	@echo "make run-phase4  : modern LLMs (vector search, RAG, fine-tuning)"
	@echo "make run-phase5  : agents (the ReAct loop)"
	@echo "make run-phase6  : engineering (evals, serving, quantization)"
	@echo "make run-all     : run every phase end to end"
	@echo "make clean       : remove caches and build output"

install:
	pip install -r requirements.txt

test: test-py test-course

test-py:
	pytest tests -q

test-course:
	cd course && npm ci --silent && npm test && npx tsc --noEmit

dev:
	cd course && npm install --silent && npm run dev

build:
	cd course && npm ci --silent && npm run build

run-phase1:
	python phase1-foundations/math_primer.py
	python phase1-foundations/gradient_descent.py
	python phase1-foundations/mlp_numpy.py

run-phase2:
	python phase2-language/bpe_tokenizer.py
	python phase2-language/tiny_word2vec.py
	python phase2-language/bigram_lm.py

run-phase3:
	python phase3-transformers/attention_numpy.py
	python phase3-transformers/tiny_gpt.py --quick
	python phase3-transformers/kv_cache_demo.py

run-phase4:
	python phase4-modern-llms/vector_db.py
	python phase4-modern-llms/mini_rag.py
	python phase4-modern-llms/finetune_tiny_gpt.py --quick

run-phase5:
	python phase5-agents/mini_agent.py

run-phase6:
	python phase6-engineering/eval_harness.py
	python phase6-engineering/batching_sim.py
	python phase6-engineering/quantize_demo.py
	python phase6-engineering/agent_budget.py

run-all: run-phase1 run-phase2 run-phase3 run-phase4 run-phase5 run-phase6

clean:
	find . -type d -name "__pycache__" -not -path "./course/node_modules/*" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache course/dist
