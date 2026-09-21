"""Make the phase folders importable (their names contain hyphens, so they are not packages)."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for folder in ("phase1-foundations", "phase2-language", "phase3-transformers",
               "phase4-modern-llms", "phase5-agents", "phase6-engineering"):
    sys.path.insert(0, os.path.join(ROOT, folder))
