# 👾 Embedding Space Invaders - Project Structure

## Core Files

### Main Code
- `embedding_space_invaders.py` - Main detection system with multi-layer embedding analysis
- `es_invaders_harness.py` - Experiment harness for running evaluations
- `run_es_invaders_experiment.py` - CLI tool for running experiments
- `metrics.py` - Statistical computation utilities

### Configuration
- `requirements.txt` - Python package dependencies
- `.gitignore` - Git ignore patterns

### Documentation
- `README.md` - Main project documentation (with humor!)
- `PROJECT_STRUCTURE.md` - This file (project overview)

## Datasets Directory

**NOT INCLUDED IN REPO** - Datasets are excluded to avoid licensing issues.

To get datasets for experiments, see the "Getting Datasets" section in README.md.

You can use:
- Quick demo mode (generates synthetic data on the fly)
- Source your own datasets from academic repositories
- Format: one prompt per line in `.txt` files

## Ignored Files (via .gitignore)

The following are automatically ignored:
- `__pycache__/` - Python cache
- `results*/` - Experiment outputs
- `logs/` - Runtime logs
- `*.jsonl` detection logs
- Virtual environments
- IDE files
- Model checkpoint files

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run quick demo
python3 run_es_invaders_experiment.py --mode quick

# Run full experiment with SEP dataset
python3 run_es_invaders_experiment.py --mode full \
    --safe-prompts datasets/safe_prompts_sep_all.txt \
    --attack-prompts datasets/attack_prompts_sep_all.txt \
    --output-dir results_sep

# Run full experiment with jailbreak dataset
python3 run_es_invaders_experiment.py --mode full \
    --safe-prompts datasets/safe_prompts_jailbreak.txt \
    --attack-prompts datasets/attack_prompts_jailbreak.txt \
    --output-dir results_jailbreak
```

## Total Size

- Code: ~90 KB (Python files only)
- Datasets: NOT INCLUDED (you download separately)
- Total repo size: **~90 KB** (super clean!)

## Development Notes

This project is intentionally kept minimal - just the code, datasets, and documentation needed to reproduce the (spectacular) failures documented in the README.
