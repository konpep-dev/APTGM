"""
Extract key metrics from APTGM training results for paper tables/figures.
Run this after downloading results from Google Drive.

Usage:
    python extract_paper_data.py --results_dir /path/to/APTGM_Results
"""

import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def load_history(path):
    """Load history.json file."""
    with open(path) as f:
        return json.load(f)


def extract_model_summary(results_dir):
    """Extract final metrics from all models."""
    results_dir = Path(results_dir)
    
    data = []
    
    # Find all history files
    for history_path in sorted(results_dir.glob('**/history.json')):
        history = load_history(history_path)
        
        # Extract model info from path
        relative_path = history_path.relative_to(results_dir)
        model_name = relative_path.parts[0]
        
        # Basic metrics
        final_acc = history['accuracy'][-1] if history['accuracy'] else 0
        best_acc = max(history['accuracy']) if history['accuracy'] else 0
        final_loss = history['loss'][-1] if history['loss'] else 0
        num_steps = len(history['step'])
        
        entry = {
            'Model': model_name,
            'Final Accuracy': final_acc,
            'Best Accuracy': best_acc,
            'Final Loss': final_loss,
            'Steps': num_steps,
        }
        
        # Add gate metrics if present
        if 'gate_at_queries' in history:
            entry['Gate@Queries'] = history['gate_at_queries'][-1]
            entry['Gate@Filler'] = history['gate_at_filler'][-1]
            entry['Gate Gap'] = history['gate_at_queries'][-1] - history['gate_at_filler'][-1]
        
        data.append(entry)
    
    return pd.DataFrame(data)


def extract_ssm_context_sweep(results_dir):
    """Extract SSM performance across sequence lengths."""
    results_dir = Path(results_dir)
    
    seq_lengths = [64, 128, 256, 512]
    data = []
    
    for seq_len in seq_lengths:
        history_path = results_dir / f'ssm_seq{seq_len}' / 'history.json'
        if history_path.exists():
            history = load_history(history_path)
            data.append({
                'Sequence Length': seq_len,
                'Final Accuracy': history['accuracy'][-1],
                'Best Accuracy': max(history['accuracy']),
            })
    
    return pd.DataFrame(data)


def extract_aptgm_gate_analysis(results_dir):
    """Extract detailed gate behavior from APTGM."""
    results_dir = Path(results_dir)
    
    history_path = results_dir / 'aptgm_seq128' / 'history.json'
    if not history_path.exists():
        return None
    
    history = load_history(history_path)
    
    # Calculate statistics
    gate_queries = np.array(history['gate_at_queries'])
    gate_filler = np.array(history['gate_at_filler'])
    gate_kv = np.array(history['gate_at_kv'])
    
    analysis = {
        'Final Gate @ Queries': gate_queries[-1],
        'Final Gate @ Filler': gate_filler[-1],
        'Final Gate @ KV': gate_kv[-1],
        'Routing Gap (Q-F)': gate_queries[-1] - gate_filler[-1],
        'Avg Gate @ Queries (last 500)': gate_queries[-500:].mean(),
        'Avg Gate @ Filler (last 500)': gate_filler[-500:].mean(),
        'Std Gate @ Queries (last 500)': gate_queries[-500:].std(),
        'Convergence': 'Selective' if gate_queries[-1] - gate_filler[-1] > 0.1 else 'Uniform',
    }
    
    return pd.Series(analysis)


def extract_baseline_comparison(results_dir):
    """Compare APTGM with fixed-α baselines."""
    results_dir = Path(results_dir)
    
    models = {
        'APTGM': results_dir / 'aptgm_seq128' / 'history.json',
        'Falcon α=0.1': results_dir / 'falcon_seq128' / 'falcon_h1_01_seq128_history.json',
        'Falcon α=0.25': results_dir / 'falcon_seq128' / 'falcon_h1_025_seq128_history.json',
    }
    
    data = []
    for name, path in models.items():
        if path.exists():
            history = load_history(path)
            data.append({
                'Model': name,
                'Final Accuracy': history['accuracy'][-1],
                'Best Accuracy': max(history['accuracy']),
            })
    
    return pd.DataFrame(data)


def print_paper_tables(results_dir):
    """Print formatted tables for paper."""
    print("="*70)
    print("APTGM PAPER - EXTRACTED RESULTS")
    print("="*70)
    print()
    
    # Table 1: Overall summary
    print("TABLE 1: Model Performance Summary @ seq=128")
    print("-"*70)
    df_summary = extract_model_summary(results_dir)
    # Filter for seq128 models
    df_seq128 = df_summary[df_summary['Model'].str.contains('seq128')]
    print(df_seq128.to_string(index=False))
    print()
    
    # Table 2: SSM context sweep
    print("TABLE 2: SSM Performance Across Sequence Lengths")
    print("-"*70)
    df_ssm = extract_ssm_context_sweep(results_dir)
    if not df_ssm.empty:
        print(df_ssm.to_string(index=False))
        print()
        # Calculate degradation
        if len(df_ssm) >= 2:
            acc_64 = df_ssm[df_ssm['Sequence Length'] == 64]['Final Accuracy'].values[0]
            acc_512 = df_ssm[df_ssm['Sequence Length'] == 512]['Final Accuracy'].values[0]
            print(f"Degradation (64→512): {acc_64 - acc_512:.2%}")
            print(f"Status: {'✅ PASS' if acc_64 - acc_512 > 0.3 else '❌ FAIL'}")
    else:
        print("⚠️ SSM context sweep data not found")
    print()
    
    # Table 3: Gate analysis
    print("TABLE 3: APTGM Gate Behavior Analysis")
    print("-"*70)
    gate_analysis = extract_aptgm_gate_analysis(results_dir)
    if gate_analysis is not None:
        print(gate_analysis.to_string())
        print()
        # Validation
        gap = gate_analysis['Routing Gap (Q-F)']
        print(f"Gate Selectivity: {'✅ PASS' if gap > 0.1 else '❌ FAIL'} (gap={gap:.3f})")
    else:
        print("⚠️ APTGM gate data not found")
    print()
    
    # Table 4: Baseline comparison
    print("TABLE 4: APTGM vs Fixed-α Baselines")
    print("-"*70)
    df_baselines = extract_baseline_comparison(results_dir)
    if not df_baselines.empty:
        print(df_baselines.to_string(index=False))
        print()
        # Find winner
        best_model = df_baselines.loc[df_baselines['Final Accuracy'].idxmax(), 'Model']
        print(f"Best model: {best_model}")
    else:
        print("⚠️ Baseline comparison data not found")
    print()
    
    print("="*70)
    

def save_to_csv(results_dir, output_dir='.'):
    """Save all tables to CSV files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Model summary
    df = extract_model_summary(results_dir)
    df.to_csv(output_dir / 'table1_model_summary.csv', index=False)
    
    # SSM context sweep
    df = extract_ssm_context_sweep(results_dir)
    df.to_csv(output_dir / 'table2_ssm_context.csv', index=False)
    
    # Gate analysis
    series = extract_aptgm_gate_analysis(results_dir)
    if series is not None:
        series.to_csv(output_dir / 'table3_gate_analysis.csv', header=['Value'])
    
    # Baseline comparison
    df = extract_baseline_comparison(results_dir)
    df.to_csv(output_dir / 'table4_baselines.csv', index=False)
    
    print(f"\n✅ CSV files saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Extract APTGM results for paper')
    parser.add_argument('--results_dir', type=str, default='APTGM_Results',
                        help='Path to results directory')
    parser.add_argument('--output_dir', type=str, default='paper_data',
                        help='Output directory for CSV files')
    parser.add_argument('--csv', action='store_true',
                        help='Save results to CSV files')
    
    args = parser.parse_args()
    
    # Check if directory exists
    if not Path(args.results_dir).exists():
        print(f"❌ Error: Directory not found: {args.results_dir}")
        print("\nUsage:")
        print("  1. Download APTGM_Results from Google Drive")
        print("  2. Run: python extract_paper_data.py --results_dir /path/to/APTGM_Results")
        return
    
    # Print tables
    print_paper_tables(args.results_dir)
    
    # Save to CSV if requested
    if args.csv:
        save_to_csv(args.results_dir, args.output_dir)


if __name__ == '__main__':
    main()
