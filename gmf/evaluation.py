# Gated Manifold Flow - Evaluation Module
# Evaluation metrics and visualization for GMF vs LUNAR

import json
import os
import argparse
from typing import Dict, List, Optional
import numpy as np


def load_results(results_dir: str) -> Dict:
    """Load all result JSON files from a directory."""
    results = {}
    for filename in os.listdir(results_dir):
        if filename.endswith('.json'):
            filepath = os.path.join(results_dir, filename)
            with open(filepath, 'r') as f:
                results[filename] = json.load(f)
    return results


def extract_metrics(results: Dict, method: str) -> Dict[str, float]:
    """Extract key metrics from results."""
    metrics = {}
    
    # Try to find the results for the method
    for key, value in results.items():
        if method.lower() in key.lower():
            if isinstance(value, dict):
                # Extract nested metrics
                for k, v in value.items():
                    if isinstance(v, (int, float)):
                        metrics[f"{k}"] = v
                    elif isinstance(v, str) and v != "N/A":
                        try:
                            metrics[f"{k}"] = float(v)
                        except:
                            pass
            break
    
    return metrics


def compare_methods(results_dir: str, output_dir: str = None):
    """Generate comparison between GMF and LUNAR."""
    results = load_results(results_dir)
    
    if not results:
        print(f"No results found in {results_dir}")
        return None
    
    # Extract metrics for both methods
    gmf_metrics = extract_metrics(results, 'gmf')
    lunar_metrics = extract_metrics(results, 'lunar')
    
    comparison = {
        'GMF': gmf_metrics,
        'LUNAR': lunar_metrics
    }
    
    # Print comparison
    print("\n" + "=" * 80)
    print("GMF vs LUNAR Comparison")
    print("=" * 80)
    
    # Key metrics for unlearning evaluation
    key_metrics = [
        ('rouge1_recall', 'ROUGE-1 Recall (lower is better for forget)'),
        ('rougeL_recall', 'ROUGE-L Recall (lower is better for forget)'),
        ('gt_loss', 'Ground Truth Loss'),
        ('mrr', 'Mean Reciprocal Rank'),
        ('hit_rate', 'Hit Rate'),
        ('perplexity', 'Perplexity'),
    ]
    
    print(f"\n{'Metric':<35} | {'GMF':<20} | {'LUNAR':<20}")
    print("-" * 80)
    
    for metric_key, metric_name in key_metrics:
        gmf_val = gmf_metrics.get(metric_key, 'N/A')
        lunar_val = lunar_metrics.get(metric_key, 'N/A')
        
        if isinstance(gmf_val, float):
            gmf_str = f"{gmf_val:.4f}"
        else:
            gmf_str = str(gmf_val)
            
        if isinstance(lunar_val, float):
            lunar_str = f"{lunar_val:.4f}"
        else:
            lunar_str = str(lunar_val)
        
        print(f"{metric_name:<35} | {gmf_str:<20} | {lunar_str:<20}")
    
    print("=" * 80)
    
    # Save comparison to file
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        comparison_path = os.path.join(output_dir, 'comparison.json')
        with open(comparison_path, 'w') as f:
            json.dump(comparison, f, indent=4)
        print(f"\nComparison saved to {comparison_path}")
    
    return comparison


def compute_unlearning_score(metrics: Dict[str, float], forget: bool = True) -> float:
    """
    Compute an aggregate unlearning score.
    
    For forget metrics: lower is better (model has forgotten)
    For retain metrics: higher is better (model has retained)
    """
    score = 0.0
    
    # ROUGE scores: lower is better for forget
    rouge1 = metrics.get('rouge1_recall', 0.5)
    rougeL = metrics.get('rougeL_recall', 0.5)
    
    if forget:
        # Lower is better for forget
        score = (1 - rouge1) + (1 - rougeL)
    else:
        # Higher is better for retain
        score = rouge1 + rougeL
    
    return score / 2


def print_summary_table(forget_results: Dict, retain_results: Dict):
    """Print a summary table of forget and retain performance."""
    print("\n" + "=" * 80)
    print("Unlearning Performance Summary")
    print("=" * 80)
    
    # Forget performance
    print("\n--- Forget Performance (lower is better) ---")
    for method, metrics in forget_results.items():
        score = compute_unlearning_score(metrics, forget=True)
        print(f"{method}: Score = {score:.4f}")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
    
    # Retain performance
    print("\n--- Retain Performance (higher is better) ---")
    for method, metrics in retain_results.items():
        score = compute_unlearning_score(metrics, forget=False)
        print(f"{method}: Score = {score:.4f}")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
    
    print("=" * 80)


def evaluate_gmf_results(results_path: str) -> Dict:
    """
    Evaluate GMF results and compute summary statistics.
    
    Args:
        results_path: Path to the GMF results JSON file
    
    Returns:
        Dictionary with evaluation metrics
    """
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    evaluation = {
        'forget': {},
        'retain': {},
        'factual': {}
    }
    
    # Extract forget metrics
    if 'forget' in results:
        forget_data = results['forget']
        evaluation['forget'] = {
            'rouge1_recall': forget_data.get('rouge1_recall', 'N/A'),
            'rougeL_recall': forget_data.get('rougeL_recall', 'N/A'),
            'gt_loss': forget_data.get('gt_loss', 'N/A'),
            'perplexity': forget_data.get('perplexity', 'N/A'),
            'mrr': forget_data.get('mrr', 'N/A'),
            'hit_rate': forget_data.get('hit_rate', 'N/A'),
        }
    
    # Extract retain metrics
    if 'retain' in results:
        retain_data = results['retain']
        evaluation['retain'] = {
            'rouge1_recall': retain_data.get('rouge1_recall', 'N/A'),
            'rougeL_recall': retain_data.get('rougeL_recall', 'N/A'),
            'gt_loss': retain_data.get('gt_loss', 'N/A'),
            'perplexity': retain_data.get('perplexity', 'N/A'),
            'mrr': retain_data.get('mrr', 'N/A'),
            'hit_rate': retain_data.get('hit_rate', 'N/A'),
        }
    
    # Extract factual metrics
    if 'factual' in results:
        factual_data = results['factual']
        evaluation['factual'] = {
            'rouge1_recall': factual_data.get('rouge1_recall', 'N/A'),
            'rougeL_recall': factual_data.get('rougeL_recall', 'N/A'),
            'gt_loss': factual_data.get('gt_loss', 'N/A'),
            'perplexity': factual_data.get('perplexity', 'N/A'),
        }
    
    return evaluation


def main():
    parser = argparse.ArgumentParser(description='GMF Evaluation Script')
    parser.add_argument('--results_dir', type=str, default='outputs/gmf_tofu',
                        help='Directory containing result JSON files')
    parser.add_argument('--output_dir', type=str, default='outputs/gmf_eval',
                        help='Directory to save evaluation results')
    parser.add_argument('--results_file', type=str, default=None,
                        help='Path to specific results file')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Gated Manifold Flow - Evaluation")
    print("=" * 80)
    
    if args.results_file:
        # Evaluate a specific results file
        evaluation = evaluate_gmf_results(args.results_file)
        print("\nEvaluation Results:")
        print(json.dumps(evaluation, indent=4))
        
        # Save evaluation
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            eval_path = os.path.join(args.output_dir, 'evaluation.json')
            with open(eval_path, 'w') as f:
                json.dump(evaluation, f, indent=4)
            print(f"\nEvaluation saved to {eval_path}")
    else:
        # Compare methods
        comparison = compare_methods(args.results_dir, args.output_dir)
    
    print("\n" + "=" * 80)
    print("Evaluation Complete")
    print("=" * 80)


if __name__ == "__main__":
    main()