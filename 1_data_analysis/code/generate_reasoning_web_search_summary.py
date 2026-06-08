#!/usr/bin/env python3
"""
Generate summary table from reasoning_web_search_metrics data across all regions.

This script processes the reasoning_web_search_metrics.csv files from all regions
(US, UK, Europe) and creates a summary table with aggregated metrics for each
unique combination of politician, prompt_version, and region.

Author: Data Science Team
Date: January 2026
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_reasoning_metrics_data(base_path: Path) -> pd.DataFrame:
    """
    Load reasoning_web_search_metrics.csv files from all regions.

    Args:
        base_path: Path to the trial_1_reason_url directory

    Returns:
        Combined DataFrame with all regions' data
    """
    regions = ["US", "UK", "Europe"]
    all_data = []

    for region in regions:
        file_path = base_path / region / "reasoning_web_search_metrics.csv"

        if file_path.exists():
            df = pd.read_csv(file_path)

            # Ensure region column matches the directory name
            df["region"] = region
            all_data.append(df)
        else:
            print(f"Warning: File not found: {file_path}")

    if not all_data:
        raise FileNotFoundError("No reasoning_web_search_metrics.csv files found")

    combined_df = pd.concat(all_data, ignore_index=True)
    return combined_df


def generate_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate summary table with aggregated metrics.

    Args:
        df: Combined DataFrame with all regions' data

    Returns:
        Summary DataFrame with aggregated metrics
    """
    # Group by politician, prompt_version, and region
    grouped = df.groupby(["politician", "prompt_version", "region"])

    summary_data = []

    for (politician, prompt_version, region), group in grouped:
        # Count total repetitions where total_web_search is 0 (model rejected request)
        rejected_requests = len(group[group["total_web_search"] == 0])

        # Calculate medians across all repetitions
        median_empty_reasoning_and_web_search = group[
            "empty_reasoning_and_web_search"
        ].median()
        median_total_reasoning = group["total_reasoning"].median()
        median_total_web_search = group["total_web_search"].median()
        median_total_urls = group["total_urls"].median()
        median_input_tokens = group["input_tokens"].median()
        median_output_tokens = group["output_tokens"].median()
        median_total_tokens = group["total_tokens"].median()

        # Total number of repetitions
        total_repetitions = len(group)

        summary_data.append(
            {
                "politician": politician,
                "prompt_version": prompt_version,
                "region": region,
                "total_repetitions": total_repetitions,
                "rejected_requests_count": rejected_requests,
                "rejected_requests_percentage": (
                    (rejected_requests / total_repetitions * 100)
                    if total_repetitions > 0
                    else 0
                ),
                "median_empty_reasoning_and_web_search": median_empty_reasoning_and_web_search,
                "median_total_reasoning": median_total_reasoning,
                "median_total_web_search": median_total_web_search,
                "median_total_urls": median_total_urls,
                "median_input_tokens": median_input_tokens,
                "median_output_tokens": median_output_tokens,
                "median_total_tokens": median_total_tokens,
            }
        )

    summary_df = pd.DataFrame(summary_data)

    # Sort by region, politician, and prompt_version for better readability
    summary_df = summary_df.sort_values(
        ["region", "politician", "prompt_version"]
    ).reset_index(drop=True)

    return summary_df


def save_summary_table(summary_df: pd.DataFrame, output_path: Path) -> None:
    """
    Save the summary table to CSV file.

    Args:
        summary_df: Summary DataFrame to save
        output_path: Path where to save the CSV file
    """
    # Ensure the results directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    summary_df.to_csv(output_path, index=False)

    # Print basic statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print(f"Total unique combinations: {len(summary_df)}")
    print(f"Regions covered: {sorted(summary_df['region'].unique())}")
    print(f"Politicians covered: {sorted(summary_df['politician'].unique())}")
    print(f"Prompt versions covered: {sorted(summary_df['prompt_version'].unique())}")

    print(f"\nRejected requests statistics:")
    print(f"  - Total rejected requests: {summary_df['rejected_requests_count'].sum()}")
    print(
        f"  - Average rejection rate: {summary_df['rejected_requests_percentage'].mean():.2f}%"
    )
    print(
        f"  - Max rejection rate: {summary_df['rejected_requests_percentage'].max():.2f}%"
    )

    print(f"\nMedian metrics across all combinations:")
    numeric_cols = [col for col in summary_df.columns if col.startswith("median_")]
    for col in numeric_cols:
        print(f"  - {col}: {summary_df[col].median():.2f}")


def main():
    """Main function to execute the summary generation process."""
    try:
        # Define paths
        base_path = (
            Path(__file__).parent.parent.parent
            / "0_prompt_llms"
            / "result"
            / "trial_1_reason_url"
        )
        output_path = (
            Path(__file__).parent.parent
            / "results"
            / "reasoning_web_search_summary.csv"
        )

        # Load data
        df = load_reasoning_metrics_data(base_path)

        # Generate summary
        summary_df = generate_summary_table(df)

        # Save results
        save_summary_table(summary_df, output_path)

    except Exception as e:
        print(f"Error during summary generation: {str(e)}")
        raise


if __name__ == "__main__":
    main()
