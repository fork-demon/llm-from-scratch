# Contributing

The goal of this repository is to make learning how LLMs work as hands-on and
engineer-friendly as possible. Improvements are welcome.

## Two halves, two sets of rules

**`phase1-foundations/` … `phase6-engineering/`** hold the Python. It is the source of
truth: the course quotes its output, and the browser ports are tested against its
behaviour. Keep it NumPy-first, with no PyTorch before `tiny_gpt.py` in Part 7, so the
mechanics stay visible.

**`course/`** holds the interactive site. Before changing a lesson, read
`course/AUTHORING.md`. It defines the 12-section lesson structure, the writing rules and
how diagrams, interactives and coding exercises are built.

## Writing

- Write like an engineer explaining at a whiteboard, not like a paper.
- No unexplained jargon. A term is either unpacked in plain words at first use, linked to
  the glossary with `<G t="...">`, or explicitly deferred to the lesson that teaches it.
- Prefer shapes and concrete numbers over Greek letters.
- Avoid "simply", "obviously", "just" and "of course". If it were obvious, the reader
  would not be here.

## Correctness comes first

Every number in a lesson must be real. If you quote what a script prints, run the script.
If you change a claim about what happens when a parameter changes, re-run it and quote
what you actually saw.

## Checks

```bash
make test        # Python tests, then the course tests and the type check
make dev         # run the course locally at http://localhost:5173
```

A pull request should say what you improved and why. If you changed a number, say how you
measured it.
