#!/usr/bin/env python3
"""
Find and Download Public Prompt Injection Datasets

This script helps you find real-world prompt injection datasets
from public sources for proper validation.
"""

import os

print("=" * 80)
print("PUBLIC PROMPT INJECTION DATASETS")
print("=" * 80)

print("""
## 1. HuggingFace Datasets

### deepset/prompt-injections
- URL: https://huggingface.co/datasets/deepset/prompt-injections
- Size: ~1,000 prompts
- Contains: Real prompt injection attempts
- Usage:
    ```python
    from datasets import load_dataset
    dataset = load_dataset("deepset/prompt-injections")
    ```

### TrustAI/PromptInjection
- URL: https://huggingface.co/datasets/TrustAI/PromptInjection
- Size: Multiple datasets combined
- Contains: Various injection techniques

### fka/awesome-chatgpt-prompts
- URL: https://huggingface.co/datasets/fka/awesome-chatgpt-prompts
- Size: 1,000+ prompts
- Contains: High-quality safe prompts (for baseline)

## 2. GitHub Repositories

### prompt-injection-dataset
- URL: https://github.com/agencyenterprise/promptinject
- Size: 1,000+ examples
- Contains: Categorized injection attempts

### LLM Security Dataset
- URL: https://github.com/llm-attacks/llm-attacks
- Contains: Adversarial prompts and attacks

### Jailbreak Prompts
- URL: https://github.com/verazuo/jailbreak_llms
- Contains: Jailbreak techniques and examples

## 3. Research Paper Datasets

### PromptBench (Microsoft Research)
- Paper: "PromptBench: Towards Evaluating the Robustness of Large Language Models"
- Contains: Adversarial prompt evaluation benchmark

### ToxicChat
- URL: https://huggingface.co/datasets/lmsys/toxic-chat
- Contains: Toxic and adversarial conversations

## 4. How to Download

### Option A: Use HuggingFace datasets library
```bash
pip install datasets
```

```python
from datasets import load_dataset

# Load prompt injection dataset
dataset = load_dataset("deepset/prompt-injections")

# Extract prompts
safe_prompts = []
attack_prompts = []

for example in dataset['train']:
    if example['label'] == 0:  # Safe
        safe_prompts.append(example['text'])
    else:  # Attack
        attack_prompts.append(example['text'])
```

### Option B: Clone GitHub repos
```bash
git clone https://github.com/agencyenterprise/promptinject
cd promptinject
# Follow their instructions to extract data
```

## 5. Manual Curation Sources

### Collect from:
- AI Red Teaming communities (Discord, Reddit)
- Bug bounty reports (HackerOne, Bugcrowd)
- Academic papers on LLM security
- ChatGPT jailbreak communities (be careful with ethics!)

## 6. Commercial Datasets

### Garak (AI Red Teaming)
- URL: https://github.com/leondz/garak
- Contains: LLM vulnerability scanner with test prompts

### OWASP Top 10 for LLMs
- URL: https://owasp.org/www-project-top-10-for-large-language-model-applications/
- Contains: Example attacks for each vulnerability

## 7. Combine Multiple Sources

For best results, combine multiple datasets:

```python
# Example combination
final_safe = []
final_attacks = []

# Add HuggingFace data
final_safe.extend(hf_safe_prompts)
final_attacks.extend(hf_attack_prompts)

# Add GitHub data
final_attacks.extend(github_prompts)

# Add your synthetic data
final_safe.extend(generated_safe_prompts)
final_attacks.extend(generated_attack_prompts)

# Result: 1000+ safe, 200+ attacks
```

## 8. Data Quality Checklist

When selecting datasets, ensure:
- ✓ Diverse attack types (injection, jailbreak, roleplay, etc.)
- ✓ Various prompt lengths (short, medium, long)
- ✓ Different domains (technical, creative, general)
- ✓ Recent attacks (techniques evolve)
- ✓ Labels are accurate
- ✓ No duplicate prompts

## 9. Quick Start Script

Run this to download and prepare datasets:

```bash
pip install datasets
python3 download_public_datasets.py
```

(Script coming next...)
""")

print("=" * 80)
print("\nNext: Creating download script...")
