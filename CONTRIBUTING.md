# Contributing to LLMs From Scratch

Thank you for your interest in improving this curriculum! Our goal is to make learning deep learning from first principles as accessible, hands-on, and engineer-friendly as possible.

## 🛠️ How to Contribute

We welcome improvements, especially in these areas:
1. **Visualizations**: Adding Matplotlib plots or interactive widgets to the `.ipynb` notebooks.
2. **Explanations**: Simplifying math explanations, fixing typos, or converting dense paragraphs into scannable lists or `<details>` blocks.
3. **Tests**: Adding robust unit tests to the `tests/` directory to ensure math integrity.
4. **Debugging**: Expanding the `debugging-guide.md` with common errors you encountered.

## 📝 Guidelines

- **Keep it NumPy-first**: Do not introduce PyTorch or other frameworks before Module 08. The goal is raw mechanics.
- **Tone**: Write like a senior engineer explaining something at a whiteboard. Avoid overly academic hedging (e.g., "It is important to note that...") and "AI-speak" (e.g., "Let's delve into...").
- **Shapes over formulas**: When explaining matrix operations, prioritize explaining the *tensor shapes* over Greek letters.

## ⚙️ Development Setup

1. Fork the repo and clone it locally.
2. Run `make install` to set up the environment.
3. Make your changes in a new branch.
4. Run `make test` to ensure no foundational math (like backprop gradients) is broken.
5. Submit a PR with a clear summary of what you improved and why!
