"""
Process manually labeled data from the manual_label folder.

This script processes the manually labeled web search queries data by:
1. Loading CSV files for each region (EU, UK, US)
2. Applying domain normalization to create common_domain column
3. Counting domain frequencies per region
4. Saving results to CSV files in the results folder
"""

import pandas as pd
import os
from typing import Dict, List
from domain_processing import normalize_domain, save_domain_counts


def load_manual_label_data(manual_label_dir: str) -> Dict[str, pd.DataFrame]:
    """
    Load manually labeled data from CSV files for each region.
    
    Args:
        manual_label_dir (str): Path to the manual_label directory
        
    Returns:
        Dict[str, pd.DataFrame]: Dictionary mapping region names to DataFrames
    """
    region_files = {
        "US": "web_search_queries_requested_domain_US.csv",
        "UK": "web_search_queries_requested_domain_UK.csv", 
        "EU": "web_search_queries_requested_domain_EU.csv"
    }
    
    data = {}
    
    for region, filename in region_files.items():
        file_path = os.path.join(manual_label_dir, filename)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            print(f"Loaded {region} data: {len(df)} rows")
            data[region] = df
        else:
            print(f"Warning: File not found: {file_path}")
            
    return data


def process_domain_column_for_region(df: pd.DataFrame, region: str) -> pd.DataFrame:
    """
    Process the domain column for a specific region by creating a common_domain column.
    
    Args:
        df (pd.DataFrame): DataFrame containing domain column
        region (str): Region name for logging
        
    Returns:
        pd.DataFrame: DataFrame with added common_domain column
    """
    df_processed = df.copy()
    
    # Apply domain normalization to create common_domain column
    df_processed['common_domain'] = df_processed['domain'].apply(normalize_domain)
    
    # Count non-null domains before and after processing
    original_domains = df_processed['domain'].notna().sum()
    processed_domains = df_processed['common_domain'].notna().sum()
    
    print(f"{region}: {original_domains} original domains -> {processed_domains} processed domains")
    
    return df_processed


def count_domain_frequencies(df: pd.DataFrame, region: str) -> pd.DataFrame:
    """
    Count the frequency of each common_domain for a region.
    
    Args:
        df (pd.DataFrame): DataFrame with common_domain column
        region (str): Region name
        
    Returns:
        pd.DataFrame: DataFrame with domain frequency counts
    """
    # Filter out rows where common_domain is null/empty
    df_filtered = df[df['common_domain'].notna() & (df['common_domain'] != '')].copy()
    
    # Count frequencies
    domain_counts = df_filtered['common_domain'].value_counts().reset_index()
    domain_counts.columns = ['common_domain', 'frequency']
    
    # Add region column
    domain_counts['region'] = region
    
    # Reorder columns
    domain_counts = domain_counts[['region', 'common_domain', 'frequency']]
    
    print(f"{region}: {len(domain_counts)} unique domains found")
    
    return domain_counts


def save_manual_label_results(
    processed_data: Dict[str, pd.DataFrame], 
    domain_counts: Dict[str, pd.DataFrame],
    output_dir: str
) -> Dict[str, str]:
    """
    Save processed manual label data and domain counts to CSV files.
    
    Args:
        processed_data (Dict[str, pd.DataFrame]): Processed data with common_domain column
        domain_counts (Dict[str, pd.DataFrame]): Domain frequency counts per region
        output_dir (str): Output directory path
        
    Returns:
        Dict[str, str]: Dictionary mapping output types to file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    
    output_paths = {}
    
    # Save processed data with common_domain column for each region
    for region, df in processed_data.items():
        processed_path = os.path.join(output_dir, f"manual_label_processed_{region}.csv")
        df.to_csv(processed_path, index=False)
        output_paths[f"processed_{region}"] = processed_path
        
    # Save domain frequency counts for each region
    for region, df in domain_counts.items():
        counts_path = os.path.join(output_dir, f"manual_label_domain_counts_{region}.csv")
        df.to_csv(counts_path, index=False)
        output_paths[f"counts_{region}"] = counts_path
        
    # Create combined domain counts across all regions
    if domain_counts:
        combined_counts = pd.concat(domain_counts.values(), ignore_index=True)
        combined_path = os.path.join(output_dir, "manual_label_domain_counts_combined.csv")
        combined_counts.to_csv(combined_path, index=False)
        output_paths["combined_counts"] = combined_path
        
    return output_paths


def main():
    """
    Main function to process manually labeled data.
    """
    # Define paths
    base_dir = "1_data_analysis"
    manual_label_dir = os.path.join(base_dir, "manual_label")
    output_dir = os.path.join(base_dir, "results")
    
    print("=" * 60)
    print("Processing Manually Labeled Data")
    print("=" * 60)
    
    # Step 1: Load data
    print("\n1. Loading manually labeled data...")
    data = load_manual_label_data(manual_label_dir)
    
    if not data:
        print("No data files found. Exiting.")
        return
        
    # Step 2: Process domain columns
    print("\n2. Processing domain columns...")
    processed_data = {}
    domain_counts = {}
    
    for region, df in data.items():
        print(f"\nProcessing {region}...")
        
        # Process domain column to create common_domain
        processed_df = process_domain_column_for_region(df, region)
        processed_data[region] = processed_df
        
        # Count domain frequencies
        counts_df = count_domain_frequencies(processed_df, region)
        domain_counts[region] = counts_df
        
        # Show top 10 domains for this region
        print(f"\nTop 10 domains in {region}:")
        print(counts_df.head(10))
        
    # Step 3: Save results
    print("\n3. Saving results...")
    output_paths = save_manual_label_results(processed_data, domain_counts, output_dir)
    
    print(f"\n✓ Saved manual label analysis results:")
    for output_type, path in output_paths.items():
        print(f"  - {output_type}: {path}")
        
    # Step 4: Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    total_rows = sum(len(df) for df in data.values())
    total_unique_domains = len(pd.concat(domain_counts.values())['common_domain'].unique())
    
    print(f"Total rows processed: {total_rows}")
    print(f"Total unique domains across all regions: {total_unique_domains}")
    
    for region, counts_df in domain_counts.items():
        print(f"{region}: {len(counts_df)} unique domains, {counts_df['frequency'].sum()} total domain entries")


if __name__ == "__main__":
    main()