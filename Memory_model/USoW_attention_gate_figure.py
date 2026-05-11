#!/usr/bin/env python3
"""
Attention Gate vs Cognitive Resource Usage Scatter Plot
Creates a clean scatter plot for AAAI paper showing the relationship between
attention gate opens and cognitive resource usage, with outlier identification.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

# File paths
INPUT_PATH = "/home/zhyuan/Desktop/Qwen-Audio/results/attention_gate_analysis.xlsx"
OUTPUT_PATH = "/home/zhyuan/Desktop/Qwen-Audio/results/attention_gates_vs_resource_usage.pdf"

blues = ['#115699', '#0E6DB3', '#5CAAD7', '#95C6DE']
reds = ['#8E0D29', '#BB1E38', '#D35B4D', '#F6BCA9']
yellows = ['#E19D49', '#E8B547', '#EFC99B']
greens = ['#365C3B', '#4A6C4C', '#7DA47C', '#ABD0A7']


def identify_outliers(x, y, threshold=2.0):
    """Identify outliers using standardized residuals from linear regression"""
    X = x.values.reshape(-1, 1)
    reg = LinearRegression().fit(X, y)
    y_pred = reg.predict(X)

    residuals = y - y_pred
    std_residuals = np.abs(residuals) / np.std(residuals)
    outlier_mask = std_residuals > threshold

    return outlier_mask, reg, y_pred


def create_scatter_plot():
    """Create the main scatter plot with outlier analysis"""

    print("Loading attention gate analysis data...")

    try:
        df = pd.read_excel(INPUT_PATH, sheet_name='Attention_Gate_Analysis')
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    print(f"Loaded {len(df)} samples")

    # Extract variables
    gates = df['Attention_Gate_Opens']
    resource_usage = df['Processing_Efficiency_Percent']

    # Identify outliers
    outlier_mask, reg, y_pred = identify_outliers(gates, resource_usage, threshold=2.0)
    outliers = df[outlier_mask]

    # Calculate statistics
    correlation = np.corrcoef(gates, resource_usage)[0, 1]
    r_squared = reg.score(gates.values.reshape(-1, 1), resource_usage)

    print(f"\nStatistical Analysis:")
    print(f"Correlation: {correlation:.3f}")
    print(f"R²: {r_squared:.3f}")
    print(f"Outliers detected: {len(outliers)}")

    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8), facecolor='white')
    ax.set_facecolor('white')

    # Main scatter plot
    scatter = ax.scatter(gates, resource_usage,
                         alpha=0.7,
                         s=60,
                         c=resource_usage,
                         cmap='Blues',
                         edgecolors='black',
                         linewidth=0.5)

    # Highlight outliers
    if len(outliers) > 0:
        ax.scatter(outliers['Attention_Gate_Opens'],
                   outliers['Processing_Efficiency_Percent'],
                   color=reds[0],
                   s=100,
                   alpha=0.8,
                   edgecolors='darkred',
                   linewidth=2,
                   label=f'Outliers (n={len(outliers)})')

    # Regression line and confidence interval
    x_line = np.linspace(gates.min(), gates.max(), 100)
    y_line = reg.predict(x_line.reshape(-1, 1))
    ax.plot(x_line, y_line, 'r--', linewidth=2, alpha=0.8,
            label=f'Linear fit (R² = {r_squared:.3f})')

    residual_std = np.std(resource_usage - y_pred)
    ax.fill_between(x_line,
                    y_line - 1.96 * residual_std,
                    y_line + 1.96 * residual_std,
                    alpha=0.2, color=reds[-2])

    # Formatting
    ax.set_xlabel('Attention Gate Open Times', fontsize=18, fontweight='bold')
    ax.set_ylabel('Cognitive Resource Usage (%)', fontsize=18, fontweight='bold')
    ax.tick_params(axis='both', labelsize=16)  # Adjust the 'labelsize' to make the numbers larger

    ax.set_xlim(0.5, gates.max() + 0.5)
    ax.set_ylim(0, 105)
    # ax.grid(True, alpha=0.3, color='gray', linestyle='-', linewidth=0.5)

    # Add quadrant boundaries and labels
    median_gates = gates.median()
    median_resource_usage = resource_usage.median()

    # Add quadrant boundary lines
    ax.axvline(median_gates, color='gray', linestyle=':', alpha=0.6, linewidth=1)
    ax.axhline(median_resource_usage, color='gray', linestyle=':', alpha=0.6, linewidth=1)

    # Add quadrant labels
    ax.text(median_gates / 2, median_resource_usage + (105 - median_resource_usage) / 2, 'Q2',
            fontsize=14, fontweight='bold', ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    ax.text(median_gates + (gates.max() - median_gates) / 2, median_resource_usage + (105 - median_resource_usage) / 2,
            'Q4',
            fontsize=14, fontweight='bold', ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    ax.text(median_gates / 2, median_resource_usage / 2, 'Q1',
            fontsize=14, fontweight='bold', ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    ax.text(median_gates + (gates.max() - median_gates) / 2, median_resource_usage / 2, 'Q3',
            fontsize=14, fontweight='bold', ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # Colorbar and legend
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Cognitive Resource Usage (%)', fontsize=18)
    cbar.ax.tick_params(labelsize=16)  # Adjust the 'labelsize' to make the color bar numbers larger
    ax.legend(fontsize=16, loc='lower right')

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nPlot saved to: {OUTPUT_PATH}")

    # Analyze outliers
    analyze_outliers(outliers, gates, resource_usage)

    # Show quadrant statistics
    analyze_quadrants(df, gates, resource_usage, correlation)

    plt.show()


def analyze_outliers(outliers, gates, resource_usage):
    """Analyze and print outlier information"""
    print(f"\n{'=' * 60}")
    print("OUTLIER ANALYSIS")
    print(f"{'=' * 60}")

    if len(outliers) > 0:
        median_gates = gates.median()
        median_resource_usage = resource_usage.median()

        print(f"\nDetected {len(outliers)} outliers:")
        print("-" * 40)

        for _, outlier in outliers.iterrows():
            gates_val = outlier['Attention_Gate_Opens']
            resource_val = outlier['Processing_Efficiency_Percent']
            file_id = outlier['File_ID']
            change_points = outlier['Change_Points']

            if gates_val > median_gates and resource_val < median_resource_usage:
                outlier_type = "High Gates, Low Resource Usage (Dense Changes in Windows)"
            elif gates_val < median_gates and resource_val > median_resource_usage:
                outlier_type = "Low Gates, High Resource Usage (Well-spaced Changes)"
            else:
                outlier_type = "Other"

            print(f"{file_id}: {gates_val} gates, {resource_val:.1f}% resource usage")
            print(f"  Type: {outlier_type}")
            print(f"  Change points: {change_points}")
            print()
    else:
        print("No significant outliers detected.")


def analyze_quadrants(df, gates, resource_usage, correlation):
    """Analyze quadrant distribution and provide insights"""
    median_gates = gates.median()
    median_resource_usage = resource_usage.median()

    # Define quadrants
    q1 = df[(df['Attention_Gate_Opens'] <= median_gates) &
            (df['Processing_Efficiency_Percent'] <= median_resource_usage)]
    q2 = df[(df['Attention_Gate_Opens'] <= median_gates) &
            (df['Processing_Efficiency_Percent'] > median_resource_usage)]
    q3 = df[(df['Attention_Gate_Opens'] > median_gates) &
            (df['Processing_Efficiency_Percent'] <= median_resource_usage)]
    q4 = df[(df['Attention_Gate_Opens'] > median_gates) &
            (df['Processing_Efficiency_Percent'] > median_resource_usage)]

    print(f"\n{'=' * 60}")
    print("QUADRANT ANALYSIS")
    print(f"{'=' * 60}")

    print(f"Q1 - Low Gates, Low Resource Usage: {len(q1)} samples ({len(q1) / len(df) * 100:.1f}%)")
    print(f"Q2 - Low Gates, High Resource Usage: {len(q2)} samples ({len(q2) / len(df) * 100:.1f}%)")
    print(f"Q3 - High Gates, Low Resource Usage: {len(q3)} samples ({len(q3) / len(df) * 100:.1f}%)")
    print(f"Q4 - High Gates, High Resource Usage: {len(q4)} samples ({len(q4) / len(df) * 100:.1f}%)")

    print(f"\n{'=' * 60}")
    print("KEY INSIGHTS FOR AAAI PAPER")
    print(f"{'=' * 60}")

    print(f"\n🎯 Main Finding:")
    print(f"   Correlation: r = {correlation:.3f}")
    if correlation > 0.5:
        print("   MODERATE positive correlation - more gates generally mean higher resource usage")
    elif correlation > 0.3:
        print("   WEAK positive correlation - resource usage depends on temporal clustering")
    else:
        print("   WEAK correlation - resource usage is independent of gate count")

    print(f"\n🔍 Selective Processing Evidence:")
    resource_range = resource_usage.max() - resource_usage.min()
    print(f"   Resource usage ranges from {resource_usage.min():.1f}% to {resource_usage.max():.1f}%")
    print(f"   Range: {resource_range:.1f}% - demonstrates highly selective attention")

    if len(q3) > 0:
        print(f"\n⚡ Cognitive Efficiency Insight:")
        print(f"   {len(q3)} samples have high gates but low resource usage")
        print("   → Changes are dense within attention windows, not spread evenly")
        print("   → Temporal clustering matters more than frequency")


def main():
    """Main execution function"""
    print("Attention Gate vs Cognitive Resource Usage Analysis")
    print("=" * 55)
    print("Creating scatter plot for AAAI paper...")
    print("=" * 55)

    create_scatter_plot()

    print("\n" + "=" * 55)
    print("Analysis completed!")


if __name__ == "__main__":
    main()