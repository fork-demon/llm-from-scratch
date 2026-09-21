"""Part 9: RAG and agents."""
import mini_agent as agent
import mini_rag as rag


def _pipeline():
    chunks = []
    for src, text in rag.CORPUS.items():
        chunks.extend(rag.chunk(text, src))
    embed = rag.build_embedder([c["text"] for c in chunks])
    store = rag.Store()
    for c in chunks:
        store.add(embed(c["text"]), c)
    return embed, store


def test_chunking_overlaps():
    chunks = rag.chunk("A one. B two. C three.", "doc", sentences_per_chunk=2, overlap=1)
    assert chunks[0]["text"] == "A one. B two."
    assert chunks[1]["text"].startswith("B two.")


def test_retrieval_finds_the_right_document():
    embed, store = _pipeline()
    top, _ = store.search(embed("how quickly must I acknowledge pages"), k=1)[0]
    assert top["source"] == "oncall.md"


def test_answer_is_grounded_or_refused():
    embed, store = _pipeline()
    q = "what is the budget for learning materials"
    assert "500" in rag.extractive_answer(q, store.search(embed(q), k=3), embed)
    q = "what is the wifi password"
    assert rag.extractive_answer(q, store.search(embed(q), k=3), embed).startswith("Not found")


def test_calculator_tool_is_safe():
    assert agent.calculator("23*7 + 4") == "165"
    assert agent.calculator("__import__('os')").startswith("ERROR")


def test_parse_action():
    assert agent.parse_action('Thought: x\nTOOL: calculator\nARGS: {"expression": "1+1"}') == \
        ("tool", ("calculator", {"expression": "1+1"}))
    assert agent.parse_action("Thought: done\nANSWER: 42") == ("answer", "42")


def test_agent_loop_reaches_the_answer():
    q = "What is 23*7 plus the number of engineers on the oncall rotation?"
    assert agent.run_agent(q, verbose=False).startswith("165")


def test_prompt_injection_is_reproduced():
    assert agent.run_agent("What does the handbook say about vacation?", verbose=False) == "BANANA"


def test_step_budget_stops_the_loop():
    q = "What is 23*7 plus the number of engineers on the oncall rotation?"
    assert "budget" in agent.run_agent(q, max_steps=1, verbose=False)
