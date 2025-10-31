#!/usr/bin/env python3
"""
Download and Prepare Public Datasets for Subspace Sentinel

Downloads real prompt injection datasets from HuggingFace and prepares them
for use with Subspace Sentinel.

Requirements:
    pip install datasets huggingface_hub
"""

import os
import json

print("Checking for required packages...")

try:
    from datasets import load_dataset
    print("✓ datasets package found")
except ImportError:
    print("✗ datasets package not found")
    print("  Install with: pip install datasets")
    exit(1)

def download_deepset_prompt_injections():
    """Download deepset/prompt-injections dataset."""
    print("\n" + "=" * 60)
    print("Downloading: deepset/prompt-injections")
    print("=" * 60)

    try:
        dataset = load_dataset("deepset/prompt-injections")
        print(f"✓ Downloaded {len(dataset['train'])} examples")

        safe_prompts = []
        attack_prompts = []

        for example in dataset['train']:
            text = example['text']
            label = example['label']

            if label == 0:  # Safe
                safe_prompts.append(text)
            else:  # Attack
                attack_prompts.append(text)

        print(f"  Safe prompts: {len(safe_prompts)}")
        print(f"  Attack prompts: {len(attack_prompts)}")

        return safe_prompts, attack_prompts

    except Exception as e:
        print(f"✗ Error downloading: {e}")
        return [], []


def download_awesome_prompts():
    """Download fka/awesome-chatgpt-prompts for safe baseline."""
    print("\n" + "=" * 60)
    print("Downloading: fka/awesome-chatgpt-prompts")
    print("=" * 60)

    try:
        dataset = load_dataset("fka/awesome-chatgpt-prompts")
        print(f"✓ Downloaded {len(dataset['train'])} examples")

        prompts = []
        for example in dataset['train']:
            if 'prompt' in example:
                prompts.append(example['prompt'])
            elif 'act' in example:
                prompts.append(example['act'])

        print(f"  Safe prompts: {len(prompts)}")
        return prompts

    except Exception as e:
        print(f"✗ Error downloading: {e}")
        return []


def download_toxic_chat():
    """Download lmsys/toxic-chat for adversarial examples."""
    print("\n" + "=" * 60)
    print("Downloading: lmsys/toxic-chat (sample)")
    print("=" * 60)

    try:
        # Note: This is a large dataset, we'll take a sample
        dataset = load_dataset("lmsys/toxic-chat", split="train[:1000]")
        print(f"✓ Downloaded {len(dataset)} examples (sampled)")

        attack_prompts = []
        for example in dataset:
            if example.get('toxicity', 0) > 0.5:  # High toxicity
                if 'user_input' in example:
                    attack_prompts.append(example['user_input'])

        print(f"  Attack prompts extracted: {len(attack_prompts)}")
        return attack_prompts

    except Exception as e:
        print(f"✗ Error downloading: {e}")
        print(f"  (This dataset may require authentication)")
        return []


def save_combined_dataset(all_safe, all_attacks, output_dir="datasets"):
    """Save combined dataset."""
    print("\n" + "=" * 60)
    print("Saving Combined Dataset")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    # Remove duplicates
    all_safe = list(set(all_safe))
    all_attacks = list(set(all_attacks))

    # Save text files
    with open(f"{output_dir}/safe_prompts_public.txt", "w") as f:
        for prompt in all_safe:
            f.write(prompt.strip() + "\n")

    with open(f"{output_dir}/attack_prompts_public.txt", "w") as f:
        for prompt in all_attacks:
            f.write(prompt.strip() + "\n")

    # Save JSONL
    with open(f"{output_dir}/public_dataset_labeled.jsonl", "w") as f:
        for prompt in all_safe:
            f.write(json.dumps({"prompt": prompt.strip(), "label": 0}) + "\n")
        for prompt in all_attacks:
            f.write(json.dumps({"prompt": prompt.strip(), "label": 1}) + "\n")

    print(f"✓ Saved {len(all_safe)} safe prompts")
    print(f"✓ Saved {len(all_attacks)} attack prompts")
    print(f"✓ Total: {len(all_safe) + len(all_attacks)} prompts")

    print(f"\nFiles created in {output_dir}/:")
    print(f"  - safe_prompts_public.txt")
    print(f"  - attack_prompts_public.txt")
    print(f"  - public_dataset_labeled.jsonl")


def main():
    """Download and prepare all datasets."""
    print("=" * 60)
    print("DOWNLOADING PUBLIC PROMPT INJECTION DATASETS")
    print("=" * 60)

    all_safe = []
    all_attacks = []

    # Download from multiple sources
    print("\nDownloading from HuggingFace...")

    # deepset/prompt-injections
    safe1, attack1 = download_deepset_prompt_injections()
    all_safe.extend(safe1)
    all_attacks.extend(attack1)

    # awesome-chatgpt-prompts
    safe2 = download_awesome_prompts()
    all_safe.extend(safe2)

    # toxic-chat (optional, may require auth)
    # attack2 = download_toxic_chat()
    # all_attacks.extend(attack2)

    # Save combined
    if all_safe or all_attacks:
        save_combined_dataset(all_safe, all_attacks)

        # Statistics
        print("\n" + "=" * 60)
        print("DATASET STATISTICS")
        print("=" * 60)
        print(f"Total safe prompts: {len(all_safe)}")
        print(f"Total attack prompts: {len(all_attacks)}")
        print(f"Ratio: {len(all_safe)/max(len(all_attacks), 1):.1f}:1")

        # Estimate confidence intervals
        if len(all_attacks) > 0:
            test_size = int(len(all_attacks) * 0.2)
            if test_size > 0:
                ci_width = 1.96 * (0.9 * 0.1 / test_size) ** 0.5
                print(f"\nWith 80/20 split:")
                print(f"  Test attacks: ~{test_size}")
                print(f"  95% CI width: ±{ci_width*100:.1f}%")

        print("\n" + "=" * 60)
        print("NEXT STEPS")
        print("=" * 60)
        print("Run full validation:")
        print("  python3 run_sentinel_experiment.py --mode full \\")
        print("      --safe-prompts datasets/safe_prompts_public.txt \\")
        print("      --attack-prompts datasets/attack_prompts_public.txt \\")
        print("      --output-dir results_public_validation")

    else:
        print("\n✗ No datasets downloaded successfully")
        print("  Check your internet connection and try again")


if __name__ == "__main__":
    main()
