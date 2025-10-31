#!/usr/bin/env python3
"""
Parse SEP Dataset for Subspace Sentinel Validation

Extracts safe and attack prompts from the SEP (System Extraction Prompts) dataset
and prepares them for statistical validation experiments.

Dataset: /Users/vincentsheahan/deftokens/datasets/SEP_dataset.json
Size: 9,160 examples (far exceeds 500+ minimum for statistical validation)

Each example contains:
- prompt_clean: Original legitimate task (SAFE)
- prompt_instructed: Task + injected question (ATTACK)
"""

import json
import os
import random
from collections import defaultdict

# Path to SEP dataset
SEP_DATASET_PATH = "/Users/vincentsheahan/deftokens/datasets/SEP_dataset.json"
OUTPUT_DIR = "datasets"

def load_sep_dataset():
    """Load SEP dataset from JSON file."""
    print("=" * 70)
    print("LOADING SEP DATASET")
    print("=" * 70)

    with open(SEP_DATASET_PATH, 'r') as f:
        data = json.load(f)

    print(f"✓ Loaded {len(data)} examples from SEP_dataset.json")
    return data


def extract_prompts(data):
    """Extract safe and attack prompts from SEP dataset."""
    print("\n" + "=" * 70)
    print("EXTRACTING PROMPTS")
    print("=" * 70)

    safe_prompts = []
    attack_prompts = []

    # Track statistics
    types = defaultdict(int)
    insistent_count = 0

    for example in data:
        # Extract safe prompt (original task)
        if 'prompt_clean' in example and example['prompt_clean'].strip():
            safe_prompts.append(example['prompt_clean'].strip())

        # Extract attack prompt (task + injection)
        if 'prompt_instructed' in example and example['prompt_instructed'].strip():
            attack_prompts.append(example['prompt_instructed'].strip())

        # Track metadata
        if 'info' in example:
            if 'type' in example['info']:
                types[example['info']['type']] += 1
            if example['info'].get('is_insistent', False):
                insistent_count += 1

    print(f"✓ Extracted {len(safe_prompts)} safe prompts (prompt_clean)")
    print(f"✓ Extracted {len(attack_prompts)} attack prompts (prompt_instructed)")

    print(f"\nTask type distribution:")
    for task_type, count in sorted(types.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {task_type}: {count}")

    print(f"\nInsistent attacks: {insistent_count} ({insistent_count/len(data)*100:.1f}%)")

    return safe_prompts, attack_prompts


def create_balanced_dataset(safe_prompts, attack_prompts, target_safe=1000, target_attacks=500):
    """
    Create balanced dataset for validation.

    SEP dataset has equal safe and attack (9,160 each).
    We'll sample to create a more realistic 2:1 safe:attack ratio.
    """
    print("\n" + "=" * 70)
    print("CREATING BALANCED DATASET")
    print("=" * 70)

    # Remove duplicates
    safe_prompts = list(set(safe_prompts))
    attack_prompts = list(set(attack_prompts))

    print(f"After deduplication: {len(safe_prompts)} safe, {len(attack_prompts)} attacks")

    # Sample if we have more than needed
    random.seed(42)  # For reproducibility

    if len(safe_prompts) > target_safe:
        safe_prompts = random.sample(safe_prompts, target_safe)
        print(f"Sampled {target_safe} safe prompts")

    if len(attack_prompts) > target_attacks:
        attack_prompts = random.sample(attack_prompts, target_attacks)
        print(f"Sampled {target_attacks} attack prompts")

    return safe_prompts, attack_prompts


def create_train_test_split(safe_prompts, attack_prompts, test_ratio=0.2):
    """Create 80/20 train/test split."""
    print("\n" + "=" * 70)
    print("CREATING TRAIN/TEST SPLIT (80/20)")
    print("=" * 70)

    random.seed(42)

    # Shuffle
    random.shuffle(safe_prompts)
    random.shuffle(attack_prompts)

    # Split
    safe_split = int(len(safe_prompts) * (1 - test_ratio))
    attack_split = int(len(attack_prompts) * (1 - test_ratio))

    safe_train = safe_prompts[:safe_split]
    safe_test = safe_prompts[safe_split:]

    attack_train = attack_prompts[:attack_split]
    attack_test = attack_prompts[attack_split:]

    print(f"Training set:")
    print(f"  Safe: {len(safe_train)}")
    print(f"  Attacks: {len(attack_train)}")
    print(f"  Total: {len(safe_train) + len(attack_train)}")

    print(f"\nTest set:")
    print(f"  Safe: {len(safe_test)}")
    print(f"  Attacks: {len(attack_test)}")
    print(f"  Total: {len(safe_test) + len(attack_test)}")

    return {
        'train': {'safe': safe_train, 'attack': attack_train},
        'test': {'safe': safe_test, 'attack': attack_test}
    }


def save_datasets(splits, output_dir=OUTPUT_DIR):
    """Save datasets in multiple formats."""
    print("\n" + "=" * 70)
    print("SAVING DATASETS")
    print("=" * 70)

    os.makedirs(output_dir, exist_ok=True)

    # Save training data (text files)
    with open(f"{output_dir}/safe_prompts_sep_train.txt", "w") as f:
        for prompt in splits['train']['safe']:
            f.write(prompt + "\n")

    with open(f"{output_dir}/attack_prompts_sep_train.txt", "w") as f:
        for prompt in splits['train']['attack']:
            f.write(prompt + "\n")

    # Save test data (text files)
    with open(f"{output_dir}/safe_prompts_sep_test.txt", "w") as f:
        for prompt in splits['test']['safe']:
            f.write(prompt + "\n")

    with open(f"{output_dir}/attack_prompts_sep_test.txt", "w") as f:
        for prompt in splits['test']['attack']:
            f.write(prompt + "\n")

    # Save combined training data (JSONL)
    with open(f"{output_dir}/sep_train_labeled.jsonl", "w") as f:
        for prompt in splits['train']['safe']:
            f.write(json.dumps({"prompt": prompt, "label": 0}) + "\n")
        for prompt in splits['train']['attack']:
            f.write(json.dumps({"prompt": prompt, "label": 1}) + "\n")

    # Save combined test data (JSONL)
    with open(f"{output_dir}/sep_test_labeled.jsonl", "w") as f:
        for prompt in splits['test']['safe']:
            f.write(json.dumps({"prompt": prompt, "label": 0}) + "\n")
        for prompt in splits['test']['attack']:
            f.write(json.dumps({"prompt": prompt, "label": 1}) + "\n")

    # Save ALL data (for full experiment)
    all_safe = splits['train']['safe'] + splits['test']['safe']
    all_attacks = splits['train']['attack'] + splits['test']['attack']

    with open(f"{output_dir}/safe_prompts_sep_all.txt", "w") as f:
        for prompt in all_safe:
            f.write(prompt + "\n")

    with open(f"{output_dir}/attack_prompts_sep_all.txt", "w") as f:
        for prompt in all_attacks:
            f.write(prompt + "\n")

    print(f"✓ Saved training datasets:")
    print(f"  - {output_dir}/safe_prompts_sep_train.txt")
    print(f"  - {output_dir}/attack_prompts_sep_train.txt")
    print(f"  - {output_dir}/sep_train_labeled.jsonl")

    print(f"\n✓ Saved test datasets:")
    print(f"  - {output_dir}/safe_prompts_sep_test.txt")
    print(f"  - {output_dir}/attack_prompts_sep_test.txt")
    print(f"  - {output_dir}/sep_test_labeled.jsonl")

    print(f"\n✓ Saved combined datasets:")
    print(f"  - {output_dir}/safe_prompts_sep_all.txt")
    print(f"  - {output_dir}/attack_prompts_sep_all.txt")


def print_statistical_validation(splits):
    """Print statistical validation analysis."""
    print("\n" + "=" * 70)
    print("STATISTICAL VALIDATION ANALYSIS")
    print("=" * 70)

    test_attacks = len(splits['test']['attack'])
    test_safe = len(splits['test']['safe'])

    print(f"\nTest set size:")
    print(f"  Safe: {test_safe}")
    print(f"  Attacks: {test_attacks}")
    print(f"  Total: {test_safe + test_attacks}")

    # Confidence interval calculation
    # CI = 1.96 * sqrt(p(1-p)/n) for 95% confidence
    # Assuming 90% accuracy
    if test_attacks > 0:
        p = 0.9
        ci_width = 1.96 * ((p * (1 - p)) / test_attacks) ** 0.5
        print(f"\n95% Confidence Interval (assuming 90% accuracy):")
        print(f"  ±{ci_width*100:.1f}% (range: {(p-ci_width)*100:.1f}% - {(p+ci_width)*100:.1f}%)")

    print("\n" + "=" * 70)
    print("COMPARISON TO REQUIREMENTS")
    print("=" * 70)

    requirements = {
        "Minimum (initial testing)": {"safe": 400, "attacks": 100},
        "Recommended (reliable results)": {"safe": 900, "attacks": 250},
        "Ideal (publication-quality)": {"safe": 2000, "attacks": 500}
    }

    for level, reqs in requirements.items():
        safe_ok = test_safe >= reqs["safe"] * 0.2  # 20% of total for test
        attack_ok = test_attacks >= reqs["attacks"] * 0.2

        status = "✓ EXCEEDS" if (safe_ok and attack_ok) else "✗ BELOW"
        print(f"\n{level}:")
        print(f"  Required test size: {int(reqs['safe']*0.2)} safe, {int(reqs['attacks']*0.2)} attacks")
        print(f"  Our test size: {test_safe} safe, {test_attacks} attacks")
        print(f"  {status}")


def print_usage_examples():
    """Print example commands for running experiments."""
    print("\n" + "=" * 70)
    print("NEXT STEPS - RUN EXPERIMENTS")
    print("=" * 70)

    print("\n1. Quick experiment (uses subset):")
    print("   python3 run_sentinel_experiment.py --mode quick")

    print("\n2. Full experiment with SEP dataset:")
    print("   python3 run_sentinel_experiment.py --mode full \\")
    print("       --safe-prompts datasets/safe_prompts_sep_all.txt \\")
    print("       --attack-prompts datasets/attack_prompts_sep_all.txt \\")
    print("       --output-dir results_sep_validation")

    print("\n3. Evaluate on test set only:")
    print("   python3 run_sentinel_experiment.py --mode evaluate \\")
    print("       --safe-prompts datasets/safe_prompts_sep_test.txt \\")
    print("       --attack-prompts datasets/attack_prompts_sep_test.txt \\")
    print("       --output-dir results_sep_test")

    print("\n4. Run with different detection modes:")
    print("   python3 run_sentinel_experiment.py --mode full \\")
    print("       --safe-prompts datasets/safe_prompts_sep_all.txt \\")
    print("       --attack-prompts datasets/attack_prompts_sep_all.txt \\")
    print("       --detection-mode voting \\")  # or max, mean
    print("       --output-dir results_sep_voting")


def main():
    """Main execution."""
    print("=" * 70)
    print("SEP DATASET PARSER FOR SUBSPACE SENTINEL")
    print("=" * 70)
    print(f"\nDataset: {SEP_DATASET_PATH}")
    print(f"Output: {OUTPUT_DIR}/")

    # Load dataset
    data = load_sep_dataset()

    # Extract prompts
    safe_prompts, attack_prompts = extract_prompts(data)

    # Create balanced dataset (1000 safe, 500 attacks)
    # This gives us a 2:1 ratio which is more realistic than 1:1
    safe_prompts, attack_prompts = create_balanced_dataset(
        safe_prompts, attack_prompts,
        target_safe=1000,
        target_attacks=500
    )

    # Create train/test split
    splits = create_train_test_split(safe_prompts, attack_prompts)

    # Save datasets
    save_datasets(splits)

    # Print statistical analysis
    print_statistical_validation(splits)

    # Print usage examples
    print_usage_examples()

    print("\n" + "=" * 70)
    print("✓ DATASET PREPARATION COMPLETE")
    print("=" * 70)
    print("\nThe SEP dataset FAR EXCEEDS statistical validation requirements:")
    print("  • Required: 500+ safe, 100+ attacks")
    print("  • SEP dataset: 9,160 safe, 9,160 attacks (original)")
    print("  • Balanced subset: 1,000 safe, 500 attacks")
    print("  • Test set: 200 safe, 100 attacks")
    print("\nWith 100 attacks in test set, you can achieve ±6% confidence intervals!")


if __name__ == "__main__":
    main()
