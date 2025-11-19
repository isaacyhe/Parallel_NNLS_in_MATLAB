#!/usr/bin/env python3
"""
Plot NNLS Performance Metrics

This script generates performance visualization plots for NNLS solvers
across different problem sizes and implementations.

Usage:
    python plot_nnls_performance.py

Author: Parallel NNLS Team
Date: 2025
License: MIT
"""

import matplotlib.pyplot as plt
import numpy as np

def plot_time_proportion():
    """
    Plot the proportion of time spent in lsqnonneg relative to total execution time.
    This demonstrates that NNLS dominates the computational cost of MPI reconstruction.
    """
    # Problem sizes
    scales = ["21x21", "81x81", "111x111", "121x121", "131x131"]
    percentages = [72.08, 64.04, 89.13, 89.07, 89.31]

    # Create bar chart
    plt.figure(figsize=(10, 6))
    bars = plt.bar(scales, percentages, color='steelblue', width=0.6,
                   edgecolor='black', linewidth=1.0)

    # Customize plot
    plt.xlabel("System Matrix Size", fontsize=14, fontweight='bold', labelpad=10)
    plt.ylabel("NNLS Time Proportion (%)", fontsize=14, fontweight='bold', labelpad=10)
    plt.title("NNLS Solver Dominates MPI Reconstruction Time",
             fontsize=16, fontweight='bold', pad=15)
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.4, linewidth=0.8)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Add value labels on bars
    for bar, pct in zip(bars, percentages):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height + 1,
                f'{pct:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Save figure
    plt.tight_layout()
    plt.savefig("nnls_time_proportion.png", dpi=300, bbox_inches='tight')
    print("Saved: nnls_time_proportion.png")
    plt.close()


def plot_speedup_comparison():
    """
    Plot speedup comparison across different implementations and problem sizes.
    """
    # Problem sizes
    sizes = ["21x21", "81x81", "111x111", "121x121", "131x131"]
    x_pos = np.arange(len(sizes))

    # Speedup data (example values - replace with actual measurements)
    matlab_baseline = np.ones(len(sizes))  # Baseline
    openmp_8_threads = [5.1, 24.7, 28.2, 32.9, 25.0]
    cuda_fp64 = [2.1, 20.1, 23.1, 27.6, 20.5]
    cuda_fp32 = [5.7, 159, 736, 837, 995]

    # Create grouped bar chart
    width = 0.2
    fig, ax = plt.subplots(figsize=(12, 7))

    ax.bar(x_pos - 1.5*width, matlab_baseline, width, label='MATLAB Baseline',
           color='gray', edgecolor='black')
    ax.bar(x_pos - 0.5*width, openmp_8_threads, width, label='OpenMP (8 threads)',
           color='forestgreen', edgecolor='black')
    ax.bar(x_pos + 0.5*width, cuda_fp64, width, label='CUDA FP64',
           color='dodgerblue', edgecolor='black')
    ax.bar(x_pos + 1.5*width, cuda_fp32, width, label='CUDA FP32',
           color='orangered', edgecolor='black')

    # Customize plot
    ax.set_xlabel('System Matrix Size', fontsize=14, fontweight='bold')
    ax.set_ylabel('Speedup (log scale)', fontsize=14, fontweight='bold')
    ax.set_title('NNLS Solver Performance: Speedup vs MATLAB Baseline',
                fontsize=16, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(sizes)
    ax.set_yscale('log')
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(axis='y', which='both', linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.savefig("nnls_speedup_comparison.png", dpi=300, bbox_inches='tight')
    print("Saved: nnls_speedup_comparison.png")
    plt.close()


def plot_precision_accuracy():
    """
    Plot the accuracy comparison between FP32 and FP64 implementations.
    """
    sizes = ["21x21", "81x81", "111x111", "121x121", "131x131",
             "201x201", "301x301"]
    # Euclidean distance between FP32 and FP64 solutions
    errors = [2.25e-3, 4.42e-3, 5.17e-3, 5.40e-3, 5.62e-3, 6.96e-3, 8.51e-3]

    plt.figure(figsize=(10, 6))
    plt.plot(sizes, errors, marker='o', linewidth=2, markersize=8,
            color='crimson', markerfacecolor='white', markeredgewidth=2)

    plt.xlabel('System Matrix Size', fontsize=14, fontweight='bold')
    plt.ylabel('Euclidean Distance (FP32 vs FP64)', fontsize=14, fontweight='bold')
    plt.title('Numerical Accuracy: FP32 vs FP64 Reconstructions',
             fontsize=16, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.xticks(rotation=15)

    # Add horizontal line for threshold
    plt.axhline(y=0.01, color='green', linestyle='--', linewidth=1.5,
               label='Typical Clinical Threshold (0.01)')
    plt.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig("fp32_vs_fp64_accuracy.png", dpi=300, bbox_inches='tight')
    print("Saved: fp32_vs_fp64_accuracy.png")
    plt.close()


def main():
    """Main function to generate all plots."""
    print("Generating NNLS performance plots...")

    # Generate individual plots
    plot_time_proportion()
    plot_speedup_comparison()
    plot_precision_accuracy()

    print("\nAll plots generated successfully!")
    print("Files created:")
    print("  - nnls_time_proportion.png")
    print("  - nnls_speedup_comparison.png")
    print("  - fp32_vs_fp64_accuracy.png")


if __name__ == "__main__":
    main()
