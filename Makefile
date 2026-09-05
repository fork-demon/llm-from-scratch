.PHONY: help install test run-phase1 run-phase2 run-phase3 run-phase4 run-phase5 run-all clean

help:
	@echo "LLM From Scratch - Command Runner"
	@echo "---------------------------------"
	@echo "make install      : Install required python dependencies"
	@echo "make test         : Run tests and gradient verification checks"
	@echo "make run-phase1   : Run Phase 1 Foundation scripts (Math, GD, MLP)"
	@echo "make run-phase2   : Run Phase 2 Language scripts (Tokenizer, Word2Vec, Bigram)"
	@echo "make run-phase3   : Run Phase 3 Transformer scripts (Attention, TinyGPT, KV Cache)"
	@echo "make run-phase4   : Run Phase 4 Modern LLMs scripts (Vector DB, RAG, Fine-tuning)"
	@echo "make run-phase5   : Run Phase 5 Agent scripts (ReAct loop)"
	@echo "make run-all      : Run all phase scripts end-to-end"
	@echo "make clean        : Clean cached files and temp artifacts"

install:
	pip install -r requirements.txt

test:
	python -m unittest discover -s tests

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
	python phase3-transformers/tiny_gpt.py
	python phase3-transformers/kv_cache_demo.py

run-phase4:
	python phase4-modern-llms/vector_db.py
	python phase4-modern-llms/mini_rag.py
	python phase4-modern-llms/finetune_tiny_gpt.py

run-phase5:
	python phase5-agents/mini_agent.py

run-all: run-phase1 run-phase2 run-phase3 run-phase4 run-phase5

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name ".DS_Store" -delete
