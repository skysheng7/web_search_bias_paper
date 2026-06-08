"""
Domain trustworthiness summary script for LLM bias analysis.

This script processes response_citations.csv files from all regions (US, UK, Europe)
and generates a summary of how frequently each common_domain is classified as
trustworthy vs untrustworthy (and other categories).
"""

import pandas as pd
import os
from domain_processing import normalize_domain


def load_response_citations(base_path: str, region: str) -> pd.DataFrame:
    """
    Load response_citations.csv for a specific region.

    Args:
        base_path (str): Base path to the result directory
        region (str): Region name (US, UK, or Europe)

    Returns:
        pd.DataFrame: Response citations data
    """
    csv_path = os.path.join(base_path, region, "response_citations.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")

    df = pd.read_csv(csv_path)
    return df


def add_common_domain(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add common_domain column using domain normalization from domain_processing.

    Args:
        df (pd.DataFrame): DataFrame with domain column

    Returns:
        pd.DataFrame: DataFrame with added common_domain column
    """
    df = df.copy()
    df["common_domain"] = df["domain"].apply(normalize_domain)
    return df


def summarize_trustworthiness(df: pd.DataFrame, region: str) -> pd.DataFrame:
    """
    Summarize trustworthiness frequency for each common_domain in a region.

    Args:
        df (pd.DataFrame): DataFrame with common_domain and trustworthy columns
        region (str): Region name

    Returns:
        pd.DataFrame: Summary with columns:
            - region
            - common_domain
            - trustworthy_count
            - untrustworthy_count
            - cited_as_both_count
            - empty_count
            - total_count
    """
    # Group by common_domain and trustworthy classification
    grouped = (
        df.groupby(["common_domain", "trustworthy"]).size().reset_index(name="count")
    )

    # Pivot to get counts for each classification
    pivot = (
        grouped.pivot(index="common_domain", columns="trustworthy", values="count")
        .fillna(0)
        .astype(int)
    )

    # Reset index to make common_domain a column
    pivot = pivot.reset_index()

    # Ensure all expected columns exist (some regions might not have all categories)
    for col in ["trustworthy", "untrustworthy", "cited_as_both", ""]:
        if col not in pivot.columns:
            pivot[col] = 0

    # Rename columns for clarity
    column_mapping = {
        "trustworthy": "trustworthy_count",
        "untrustworthy": "untrustworthy_count",
        "cited_as_both": "cited_as_both_count",
        "": "empty_count",
    }
    pivot = pivot.rename(columns=column_mapping)

    # Calculate total count
    count_columns = [
        "trustworthy_count",
        "untrustworthy_count",
        "cited_as_both_count",
        "empty_count",
    ]
    pivot["total_count"] = pivot[count_columns].sum(axis=1)

    # Add region column
    pivot.insert(0, "region", region)

    # Sort by total count descending
    pivot = pivot.sort_values("total_count", ascending=False)

    return pivot


def process_all_regions(base_path: str, regions: list) -> pd.DataFrame:
    """
    Process all regions and combine results into a single DataFrame.

    Args:
        base_path (str): Base path to the result directory
        regions (list): List of region names

    Returns:
        pd.DataFrame: Combined summary for all regions
    """
    all_summaries = []

    for region in regions:
        print(f"\nProcessing {region}...")

        # Load data
        df = load_response_citations(base_path, region)
        print(f"  Loaded {len(df)} citations")

        # Add common_domain column
        df = add_common_domain(df)
        print(f"  Added common_domain column")

        # Summarize trustworthiness
        summary = summarize_trustworthiness(df, region)
        print(f"  Found {len(summary)} unique common domains")

        all_summaries.append(summary)

    # Combine all summaries
    combined = pd.concat(all_summaries, ignore_index=True)

    return combined


def main():
    """
    Main function to process all regions and save results.
    """
    # Configuration
    base_path = "0_prompt_llms/result/trial_1_reason_url"
    output_dir = "1_data_analysis/results"
    output_file = "domain_trustworthiness_summary.csv"
    regions = ["US", "UK", "Europe"]

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Process all regions
    combined_summary = process_all_regions(base_path, regions)

    # Save results
    output_path = os.path.join(output_dir, output_file)
    combined_summary.to_csv(output_path, index=False)

    print(f"\n✓ Results saved to: {output_path}")
    print(f"  Total rows: {len(combined_summary)}")


if __name__ == "__main__":
    main()
