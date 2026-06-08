"""
Script to create bar plots for reasoning and web search metrics by region.
Creates side-by-side bar plots for US, UK, and Europe regions.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_and_prepare_data(csv_path):
    """Load CSV data keeping prompt versions separate."""
    df = pd.read_csv(csv_path)
    return df


def create_bar_plot_grid(df, metric_col, title, output_path):
    """
    Create a grid of 3 bar plots (one per region) for a given metric.
    Each politician has two bars: one for 'disregards' and one for 'values'.

    Parameters:
    -----------
    df : DataFrame
        Data containing politician, region, prompt_version, and metric columns
    metric_col : str
        Column name of the metric to plot
    title : str
        Main title for the figure
    output_path : str or Path
        Path to save the output PNG file
    """
    regions = ["US", "UK", "Europe"]
    fig, axes = plt.subplots(1, 3, figsize=(24, 12))

    # Define colors for each prompt version
    colors = {"disregards": "lightcoral", "values": "lightskyblue"}

    # Store legend handles and labels for figure-level legend
    legend_handles = []
    legend_labels = []

    for idx, region in enumerate(regions):
        ax = axes[idx]

        # Filter data for this region
        region_data = df[df["region"] == region].copy()

        # Get unique politicians and sort by 'values' prompt version metric value
        values_data = region_data[region_data["prompt_version"] == "values"]
        politician_avg = (
            values_data.groupby("politician")[metric_col].mean().sort_values()
        )
        politicians_sorted = politician_avg.index.tolist()

        # Prepare data for grouped bars
        bar_height = 0.35
        y_positions = np.arange(len(politicians_sorted))

        # Plot bars for each prompt version
        for i, politician in enumerate(politicians_sorted):
            politician_data = region_data[region_data["politician"] == politician]

            for prompt_version in ["disregards", "values"]:
                data_point = politician_data[
                    politician_data["prompt_version"] == prompt_version
                ]

                if not data_point.empty:
                    value = data_point[metric_col].values[0]
                    # Offset bars vertically
                    y_pos = (
                        i - bar_height / 2
                        if prompt_version == "disregards"
                        else i + bar_height / 2
                    )

                    bar = ax.barh(
                        y_pos,
                        value,
                        height=bar_height,
                        color=colors[prompt_version],
                        alpha=0.85,
                    )

                    # Capture legend handles from first region only
                    if idx == 0 and i == 0:
                        legend_handles.append(bar)
                        legend_labels.append(prompt_version)

                    # Add value labels on bars
                    ax.text(
                        value + 0.5,
                        y_pos,
                        f"{value:.1f}",
                        va="center",
                        fontsize=16,
                    )

        # Set labels
        ax.set_yticks(y_positions)
        ax.set_yticklabels(politicians_sorted, fontsize=20)
        ax.set_title(region, fontsize=28, fontweight="bold")

        # Set x-axis limits to prevent label overflow
        max_value = region_data[metric_col].max()
        ax.set_xlim(0, max_value * 1.265)

        # Set tick label font sizes
        ax.tick_params(axis="x", labelsize=18)

        # Grid for better readability
        ax.grid(axis="x", alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)

    # Add figure-level legend at top left of entire canvas
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper left",
        fontsize=18,
        framealpha=0.9,
        title="Prompt Version",
        title_fontsize=20,
        bbox_to_anchor=(0.02, 0.98),
    )

    # Main title at bottom center with more space
    fig.suptitle(title, fontsize=32, fontweight="bold", y=0.04)
    plt.tight_layout(rect=[0, 0.08, 1, 0.96])

    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def main():
    # Define paths
    project_root = Path(__file__).parent.parent.parent
    data_path = (
        project_root / "1_data_analysis" / "results" / "reasoning_web_search_summary.csv"
    )
    output_dir = project_root / "1_data_analysis" / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and prepare data
    print("Loading data...")
    df = load_and_prepare_data(data_path)

    # Define metrics to plot
    metrics = [
        (
            "median_empty_reasoning_and_web_search",
            "Median Empty Reasoning and Web Search by Region",
        ),
        ("median_total_reasoning", "Median Total Reasoning by Region"),
        ("median_total_web_search", "Median Total Web Search by Region"),
        ("median_total_urls", "Median Total URLs by Region"),
        ("median_input_tokens", "Median Input Tokens by Region"),
        ("median_output_tokens", "Median Output Tokens by Region"),
        ("median_total_tokens", "Median Total Tokens by Region"),
    ]

    # Create plots for each metric
    for metric_col, title in metrics:
        output_filename = f"{metric_col}_bar_plot.png"
        output_path = output_dir / output_filename

        print(f"\nCreating plot for: {metric_col}")
        create_bar_plot_grid(df, metric_col, title, output_path)

    print("\n✓ All plots generated successfully!")


if __name__ == "__main__":
    main()
