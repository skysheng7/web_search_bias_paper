"""
Script to combine reasoning_web_search_metrics from US, UK, and Europe regions
and calculate averages for total_reasoning, total_web_search, and total_urls.
"""

import pandas as pd
from pathlib import Path


def combine_regional_metrics(base_path: str) -> pd.DataFrame:
    """
    Combine reasoning_web_search_metrics.csv from US, UK, and Europe regions.

    Parameters:
    -----------
    base_path : str
        Base path to the trial_1_reason_url directory

    Returns:
    --------
    pd.DataFrame
        Combined dataframe with all regional data
    """
    regions = ["US", "UK", "Europe"]
    dataframes = []

    for region in regions:
        file_path = Path(base_path) / region / "reasoning_web_search_metrics.csv"
        if file_path.exists():
            df = pd.read_csv(file_path)
            print(f"Loaded {region}: {len(df)} rows")
            dataframes.append(df)
        else:
            print(f"Warning: File not found for {region}: {file_path}")

    if not dataframes:
        raise FileNotFoundError("No regional data files found")

    combined_df = pd.concat(dataframes, ignore_index=True)
    print(f"\nCombined total: {len(combined_df)} rows")

    return combined_df


def calculate_averages(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Calculate averages for specified columns across all regions.

    Parameters:
    -----------
    df : pd.DataFrame
        Combined dataframe
    columns : list
        List of column names to calculate averages for

    Returns:
    --------
    pd.DataFrame
        DataFrame with average statistics
    """
    # Overall averages across all regions
    overall_averages = df[columns].mean()

    # Averages by region
    region_averages = df.groupby("region")[columns].mean()

    results = {
        "metric": columns,
        "overall_average": [overall_averages[col] for col in columns],
    }

    # Add regional averages
    for region in df["region"].unique():
        results[f"{region}_average"] = [
            region_averages.loc[region, col] for col in columns
        ]

    return pd.DataFrame(results)


def calculate_sums(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Calculate sums for specified columns across all regions.

    Parameters:
    -----------
    df : pd.DataFrame
        Combined dataframe
    columns : list
        List of column names to calculate sums for

    Returns:
    --------
    pd.DataFrame
        DataFrame with sum statistics
    """
    # Overall sums across all regions
    overall_sums = df[columns].sum()

    # Sums by region
    region_sums = df.groupby("region")[columns].sum()

    results = {
        "metric": columns,
        "overall_sum": [overall_sums[col] for col in columns],
    }

    # Add regional sums
    for region in df["region"].unique():
        results[f"{region}_sum"] = [region_sums.loc[region, col] for col in columns]

    return pd.DataFrame(results)


def count_response_citations(base_path: str) -> dict:
    """
    Count rows in response_citations.csv for each region.

    Parameters:
    -----------
    base_path : str
        Base path to the trial_1_reason_url directory

    Returns:
    --------
    dict
        Dictionary with row counts per region
    """
    regions = ["US", "UK", "Europe"]
    counts = {}

    for region in regions:
        file_path = Path(base_path) / region / "response_citations.csv"
        if file_path.exists():
            df = pd.read_csv(file_path)
            counts[region] = len(df)
            print(f"{region}: {len(df)} rows")
        else:
            print(f"Warning: File not found for {region}: {file_path}")
            counts[region] = 0

    counts["Total"] = sum(counts.values())
    return counts


def main():
    """Main execution function."""
    # Define base path
    base_path = Path(__file__).parent.parent / "result" / "trial_1_reason_url"

    # Combine regional data
    print("=" * 60)
    print("Combining regional metrics...")
    print("=" * 60)
    combined_df = combine_regional_metrics(base_path)

    # Calculate averages and sums
    print("\n" + "=" * 60)
    print("Calculating averages and sums...")
    print("=" * 60)
    columns_to_analyze = [
        "total_reasoning",
        "total_web_search",
        "total_urls",
        "total_tokens",
    ]
    averages_df = calculate_averages(combined_df, columns_to_analyze)
    sums_df = calculate_sums(combined_df, columns_to_analyze)

    # Display average results
    print("\n" + "=" * 60)
    print("RESULTS: Average across all 3 regions")
    print("=" * 60)
    for _, row in averages_df.iterrows():
        print(f"{row['metric']:<25} {row['overall_average']:>10.2f}")

    print("\n" + "=" * 60)
    print("RESULTS: Average by region")
    print("=" * 60)
    print(averages_df.to_string(index=False))

    # Display sum results
    print("\n" + "=" * 60)
    print("RESULTS: Sum across all 3 regions")
    print("=" * 60)
    for _, row in sums_df.iterrows():
        print(f"{row['metric']:<25} {row['overall_sum']:>10.0f}")

    print("\n" + "=" * 60)
    print("RESULTS: Sum by region")
    print("=" * 60)
    print(sums_df.to_string(index=False))

    # Additional statistics
    print("\n" + "=" * 60)
    print("Additional statistics")
    print("=" * 60)
    print(f"Total number of prompts analyzed: {len(combined_df)}")
    print(f"Regions included: {', '.join(combined_df['region'].unique())}")
    print(f"Prompts per region:")
    for region, count in combined_df["region"].value_counts().items():
        print(f"  {region}: {count}")

    # Count response_citations rows
    print("\n" + "=" * 60)
    print("Response Citations Row Counts")
    print("=" * 60)
    citation_counts = count_response_citations(base_path)
    print(f"\nTotal rows across all regions: {citation_counts['Total']}")


if __name__ == "__main__":
    main()
