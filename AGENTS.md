# AGENTS.md

## Project Overview

TakeMeter: a fine-tuned text classifier that evaluates discourse quality on `r/LocalLLaMA`. Three-label taxonomy (High-Signal Technical, Low-Effort Noise, News & Meta Discourse). Uses DSPy with DeepSeek for classification, with plans to fine-tune DistilBERT on the annotated dataset.

## Build / Run / Test Commands

```bash
# Run the main script (placeholder)
uv run python main.py

# Run the DSPy classification pipeline (reads posts from LocalLlama_Posts_with_Commentary/)
uv run python classify_posts.py

# Launch Jupyter for data exploration
uv run jupyter lab

# Add a dependency
uv add <package>

# Sync dependencies from lockfile
uv sync
```

There are currently **no tests**, **no linter**, and **no type checker** configured. If you add one, document it here.

## Project Structure

```
.
├── planning.md                          # Label taxonomy, evaluation plan, edge case docs
├── README.md                            # (empty — fill in per rubric)
├── pyproject.toml                       # uv project config, dependencies
├── main.py                              # Placeholder entrypoint
├── classify_posts.py                    # DSPy classification pipeline
├── data.ipynb                           # Exploration notebook (LLM prompt building)
├── classification_results.json          # Output of classify_posts.py
├── .env                                 # API keys — NEVER commit
├── LocalLlama_Posts_with_Commentary/    # Raw post data (first line = commentary, blank, then body)
│   ├── 1.txt, 2.txt, ...               # Individual posts (some have .txt, some don't)
├── logs/                                # Classification run logs
├── backups/                             # Backup of classification_result snapshots
└── .venv/                               # Virtual environment (gitignored)
```

## Data Format

Each file in `LocalLlama_Posts_with_Commentary/`:
- **Line 1**: Human commentary / subjective label hint
- **Line 2**: Empty line
- **Lines 3+**: Full post content (Reddit title + body)

The `classify_posts.py` script parses this format and uses both the commentary and content for DSPy classification.

## Code Style

### Imports
- Standard library first, then third-party, then local
- Use `pathlib.Path` for all file paths (never `os.path` or string concatenation)
- Import specific names, not wildcards

```python
# Correct
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

# Avoid
import os.path
from tqdm import *
```

### Formatting
- Use the project's `.py` files as reference style
- Consistent spacing around operators and after commas
- Keep lines under ~120 characters where practical

### Types
- Type hints are not currently used but encouraged for new code
- Use basic annotations: `str`, `int`, `list[dict]`, `Path`

### Naming
- `snake_case` for variables, functions, methods
- `PascalCase` for classes (`LocalLlamaPostClassifier`, `PostClassifier`)
- `UPPER_CASE` for module-level constants
- Descriptive names over abbreviations

### Error Handling
- Currently uses `assert` for data integrity checks (e.g., blank line between commentary and body)
- For new code: prefer explicit `raise ValueError(...)` with descriptive messages over bare assertions
- Wrap external API calls in try/except

### DSPy Patterns
- Signatures are defined as inner classes extending `dspy.Signature`
- Modules extend `dspy.Module` with a `forward()` method
- LLM configuration uses `dspy.LM` with OpenAI-compatible API base
- ChainOfThought wrapping is used for reasoning-capable classification

### Jupyter Notebooks
- Use `tqdm.notebook.tqdm` (not `tqdm.tqdm`) inside notebooks
- Keep notebooks focused on exploration; move production logic to `.py` files

## Environment & API Keys

- API keys are stored in `.env` (gitignored)
- Load with `load_dotenv()` at module level
- Currently uses DeepSeek API via OpenAI-compatible base URL
- Format:
  ```
  OPENAI_API_BASE="https://api.deepseek.com"
  OPENAI_API_KEY="sk-..."
  OPENAI_MODEL="deepseek-v4-pro"
  ```
- **NEVER commit `.env` or hardcode API keys in source files**

## Label Taxonomy (for reference)

| Label | Purpose |
|-------|---------|
| High-Signal Technical | Novel technical value, detailed projects, specific benchmarking/troubleshooting with prior research |
| Low-Effort Noise | Repetitive beginner questions, clickbait, social media screenshots, vibecoded slop |
| News & Meta Discourse | Legitimate model releases, industry news impacting open-source, high-effort community opinion |

Decision boundary for ambiguous posts: **demonstrated effort and specificity** is the tiebreaker. If still 50/50, default to Low-Effort Noise to protect feed quality.

## Git Workflow

- Commit messages should be concise and descriptive
- Keep `.env` and `backups/` out of commits (both gitignored)
- Don't commit generated outputs (`classification_results.json`, `logs/`) unless explicitly needed

## Relevant Project Context

- This is a CodePath Week 3 project (AI-201 / CS4650)
- Target model for fine-tuning: `distilbert-base-uncased` on Google Colab T4 GPU
- Baseline comparison model: Groq's `llama-3.3-70b-versatile` (zero-shot)
- The project rubric requires: label taxonomy docs, 200+ annotated examples, fine-tuning pipeline, baseline comparison, and evaluation report
