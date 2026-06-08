"""
Domain processing utilities for LLM bias analysis visualization.

This module provides functions to clean domain names and process web search sources
and response citations data for Sankey chart visualization.
"""

import pandas as pd
from urllib.parse import urlparse
from typing import Dict, List, Tuple
import re


def clean_domain(domain: str) -> str:
    """
    Remove 'www.' prefix from domain names to get the real domain.

    Args:
        domain (str): Domain name potentially containing 'www.' prefix

    Returns:
        str: Cleaned domain name without 'www.' prefix

    Examples:
        >>> clean_domain('www.nytimes.com')
        'nytimes.com'
        >>> clean_domain('nytimes.com')
        'nytimes.com'
        >>> clean_domain('www.bbc.co.uk')
        'bbc.co.uk'
    """
    if pd.isna(domain):
        return domain

    domain = str(domain).strip()

    # Remove www. prefix (case insensitive)
    if domain.lower().startswith("www."):
        return domain[4:]

    return domain


def process_domain_column(df: pd.DataFrame, domain_col: str = "domain") -> pd.DataFrame:
    """
    Process the domain column in a DataFrame to remove 'www.' prefixes.

    Args:
        df (pd.DataFrame): DataFrame containing domain column
        domain_col (str): Name of the domain column (default: 'domain')

    Returns:
        pd.DataFrame: DataFrame with cleaned domain column
    """
    df = df.copy()
    df[domain_col] = df[domain_col].apply(clean_domain)
    return df


def calculate_domain_counts_by_group(
    df: pd.DataFrame, region: str, domain_col: str = "domain"
) -> pd.DataFrame:
    """
    Calculate domain counts for each region, politician, and prompt_version combination.

    Args:
        df (pd.DataFrame): DataFrame with cleaned domains
        region (str): Region name (e.g., 'US', 'UK', 'Europe')
        domain_col (str): Name of the domain column

    Returns:
        pd.DataFrame: Aggregated domain counts with columns:
            - region
            - politician
            - prompt_version
            - domain
            - count
    """
    # Group by region, politician, prompt_version, and domain
    grouped = (
        df.groupby(["region", "politician", "prompt_version", domain_col])
        .size()
        .reset_index(name="count")
    )

    # Filter for the specified region
    grouped = grouped[grouped["region"] == region].copy()

    # Sort by count descending
    grouped = grouped.sort_values(
        ["politician", "prompt_version", "count"], ascending=[True, True, False]
    )

    return grouped


def calculate_total_domain_counts(
    df: pd.DataFrame, region: str, domain_col: str = "domain"
) -> pd.DataFrame:
    """
    Calculate total domain counts for an entire region (aggregated across all politicians and prompt versions).

    Args:
        df (pd.DataFrame): DataFrame with cleaned domains
        region (str): Region name (e.g., 'US', 'UK', 'Europe')
        domain_col (str): Name of the domain column

    Returns:
        pd.DataFrame: Total domain counts with columns:
            - region
            - domain
            - total_count
    """
    # Filter for the specified region
    region_df = df[df["region"] == region].copy()

    # Group by domain and count
    total_counts = region_df.groupby(domain_col).size().reset_index(name="total_count")

    # Add region column
    total_counts["region"] = region

    # Sort by count descending
    total_counts = total_counts.sort_values("total_count", ascending=False)

    # Reorder columns
    total_counts = total_counts[["region", domain_col, "total_count"]]

    return total_counts


def process_csv_file(
    csv_path: str, region: str, domain_col: str = "domain"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Process a CSV file (web_search_sources or response_citations) and return:
    1. Domain counts by group (politician, prompt_version)
    2. Total domain counts for the region
    3. Updated DataFrame with common_domain column

    Args:
        csv_path (str): Path to the CSV file
        region (str): Region name (e.g., 'US', 'UK', 'Europe')
        domain_col (str): Name of the domain column

    Returns:
        Tuple containing:
            - Domain counts by group
            - Total domain counts
            - Updated DataFrame with common_domain column
    """
    # Read CSV
    df = pd.read_csv(csv_path)

    # Clean domains (remove www. prefix)
    df_cleaned = df.copy()

    # Add normalized common_domain column
    df_cleaned["common_domain"] = df_cleaned[domain_col].apply(normalize_domain)

    # Calculate counts by group using common_domain
    group_counts = calculate_domain_counts_by_group(df_cleaned, region, "common_domain")

    # Calculate total counts using common_domain
    total_counts = calculate_total_domain_counts(df_cleaned, region, "common_domain")

    return group_counts, total_counts, df_cleaned


def get_multipart_tlds() -> List[str]:
    """
    Get list of multi-part top-level domains that require extracting more than 2 parts.

    Returns:
        List[str]: List of multi-part TLDs
    """
    return [
        "com.tr",  # Turkey
        "co.uk",  # United Kingdom
        "org.uk",  # United Kingdom
    ]


def should_extract_extra_parts(domain: str) -> int:
    """
    Determine how many parts to extract from a domain based on its TLD.

    Args:
        domain (str): Domain name

    Returns:
        int: Number of parts to extract (2 for standard domains, 3 for multi-part TLDs)
    """
    domain_lower = domain.lower()
    multipart_tlds = get_multipart_tlds()

    for tld in multipart_tlds:
        if domain_lower.endswith(tld):
            return 3

    return 2


def extract_common_domain(domain: str) -> str:
    """
    Extract the domain theme by taking the last 2 or 3 parts of the domain name,
    depending on whether it has a multi-part TLD (com.tr, co.uk, org.uk, etc.).
    For example: 'congress.gov' -> 'congress.gov', 'warren.senate.gov' -> 'senate.gov'

    Args:
        domain (str): Domain name

    Returns:
        str: Domain theme (last 2 or 3 parts of domain depending on TLD)

    Examples:
        >>> extract_common_domain('warren.senate.gov')
        'senate.gov'
        >>> extract_common_domain('congress.gov')
        'congress.gov'
        >>> extract_common_domain('www.nytimes.com')
        'nytimes.com'
        >>> extract_common_domain('emergency.unhcr.org')
        'unhcr.org'
        >>> extract_common_domain('subdomain.example.com.tr')
        'example.com.tr'
        >>> extract_common_domain('news.bbc.co.uk')
        'bbc.co.uk'
        >>> extract_common_domain('subdomain.amnesty.org.uk')
        'amnesty.org.uk'
    """
    if pd.isna(domain):
        return domain

    domain = str(domain).strip()
    parts = domain.split(".")

    # If domain has 2 or fewer parts, return as is
    if len(parts) <= 2:
        return domain

    # Determine how many parts to extract based on TLD
    parts_to_extract = should_extract_extra_parts(domain)

    # Make sure we don't try to extract more parts than available
    if len(parts) >= parts_to_extract:
        return ".".join(parts[-parts_to_extract:])

    # Fallback: return the whole domain if it has fewer parts than expected
    return domain


def apply_pattern_based_normalization(domain: str) -> str:
    """
    Apply pattern-based normalization for specific domain keywords.

    Rules:
    - Any domain containing 'amnesty' -> 'amnesty*'
    - Any domain containing 'wikipedia' -> 'wikipedia*'

    Args:
        domain (str): Domain name (should be lowercase and stripped)

    Returns:
        str: Normalized domain if pattern matches, otherwise returns original domain

    Examples:
        >>> apply_pattern_based_normalization('amnesty.org')
        'amnesty*'
        >>> apply_pattern_based_normalization('amnestyusa.org')
        'amnesty*'
        >>> apply_pattern_based_normalization('en.wikipedia.org')
        'wikipedia*'
        >>> apply_pattern_based_normalization('congress.gov')
        'congress.gov'
    """
    if "amnesty" in domain:
        return "amnesty*"

    if "wikipedia" in domain:
        return "wikipedia*"

    return domain


def get_default_domain_mapping() -> Dict[str, str]:
    """
    Get the default domain normalization mapping.
    This maps variant domains to their canonical forms.

    Note: Pattern-based matching (amnesty*, wikipedia*) is handled separately
    in apply_pattern_based_normalization().

    Returns:
        Dict[str, str]: Dictionary mapping variant domains to canonical domains
    """
    return {
        # UN variations - exact matches
        "un-ilibrary.org": "un.org",
        "ungeneva.org": "un.org",
        # Add more exact mappings as needed
    }


def normalize_domain(domain: str, domain_mapping: Dict[str, str] = None) -> str:
    """
    Normalize domain names by mapping related domains to a canonical form.
    Supports both exact matches and pattern-based matching.

    Process order:
    1. Clean domain (remove www. prefix)
    2. Extract common domain (last 2 parts)
    3. Check exact matches in domain_mapping
    4. Apply pattern-based normalization

    Args:
        domain (str): Domain name to normalize
        domain_mapping (Dict[str, str]): Dictionary mapping variant domains to canonical domains.
                                         If None, uses default mapping.

    Returns:
        str: Normalized domain name

    Examples:
        >>> normalize_domain('www.amnestyusa.org')
        'amnesty*'
        >>> normalize_domain('en.wikipedia.org')
        'wikipedia*'
        >>> normalize_domain('www.ungeneva.org')
        'un.org'
        >>> normalize_domain('warren.senate.gov')
        'senate.gov'
    """
    if pd.isna(domain):
        return domain

    # Step 1: Clean domain (remove www. prefix)
    cleaned_domain = clean_domain(domain)

    clean_domain_lower = str(cleaned_domain).strip().lower()

    # Step 2: Check if domain is in exact mapping
    # Default mapping if none provided
    if domain_mapping is None:
        domain_mapping = get_default_domain_mapping()

    if clean_domain_lower in domain_mapping:
        return domain_mapping[clean_domain_lower]

    # Step 3: Extract common domain (last 2 parts)
    clean_domain_lower_processed = extract_common_domain(clean_domain_lower)

    # Step 4: Apply pattern-based normalization
    return apply_pattern_based_normalization(clean_domain_lower_processed)


def add_common_domain_column(
    df: pd.DataFrame, domain_col: str = "domain", theme_col: str = "common_domain"
) -> pd.DataFrame:
    """
    Add a common_domain column to the DataFrame by extracting the last 2 parts of each domain.

    Args:
        df (pd.DataFrame): DataFrame containing domain column
        domain_col (str): Name of the domain column (default: 'domain')
        theme_col (str): Name of the new theme column (default: 'common_domain')

    Returns:
        pd.DataFrame: DataFrame with added common_domain column
    """
    df = df.copy()
    df[theme_col] = df[domain_col].apply(extract_common_domain)
    return df


def normalize_domain_column(
    df: pd.DataFrame, domain_col: str = "domain", domain_mapping: Dict[str, str] = None
) -> pd.DataFrame:
    """
    Normalize domain names in a DataFrame using the provided mapping.

    Args:
        df (pd.DataFrame): DataFrame containing domain column
        domain_col (str): Name of the domain column (default: 'domain')
        domain_mapping (Dict[str, str]): Custom domain mapping. If None, uses default.

    Returns:
        pd.DataFrame: DataFrame with normalized domain column
    """
    df = df.copy()
    df[domain_col] = df[domain_col].apply(lambda x: normalize_domain(x, domain_mapping))
    return df


def get_top_n_domains(
    df: pd.DataFrame, n: int = 20, count_col: str = "total_count"
) -> pd.DataFrame:
    """
    Get the top N domains by count.

    Args:
        df (pd.DataFrame): DataFrame with domain counts
        n (int): Number of top domains to return
        count_col (str): Name of the count column

    Returns:
        pd.DataFrame: Top N domains
    """
    return df.nlargest(n, count_col)


def save_domain_counts(
    group_counts: pd.DataFrame,
    total_counts: pd.DataFrame,
    region: str,
    output_dir: str,
    file_prefix: str = "citations",
    updated_df: pd.DataFrame = None,
) -> Dict[str, str]:
    """
    Save domain counts to CSV files in the output directory.

    Args:
        group_counts (pd.DataFrame): Domain counts by group (politician, prompt_version)
        total_counts (pd.DataFrame): Total domain counts
        region (str): Region name (e.g., 'US', 'UK', 'Europe')
        output_dir (str): Directory path to save the CSV files
        file_prefix (str): Prefix for output files (default: 'citations')
        updated_df (pd.DataFrame, optional): Updated DataFrame with common_domain column

    Returns:
        Dict[str, str]: Dictionary mapping output types to file paths

    Saves the following files:
        - {file_prefix}_group_counts_{region}.csv: Domain counts by group
        - {file_prefix}_total_counts_{region}.csv: Total domain counts
        - {data_type}_with_common_domain_{region}.csv: Updated data with common_domain column (if provided)
    """
    import os

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    output_paths = {}

    # 1. Save group counts
    group_path = os.path.join(output_dir, f"{file_prefix}_group_counts_{region}.csv")
    group_counts.to_csv(group_path, index=False)
    output_paths["group_counts"] = group_path

    # 2. Save total counts
    total_path = os.path.join(output_dir, f"{file_prefix}_total_counts_{region}.csv")
    total_counts.to_csv(total_path, index=False)
    output_paths["total_counts"] = total_path

    # 3. Save updated DataFrame with common_domain column (if provided)
    if updated_df is not None:
        # Determine data type from file_prefix
        data_type = (
            "web_search_sources" if file_prefix == "search" else "response_citations"
        )
        updated_path = os.path.join(
            output_dir, f"{data_type}_with_common_domain_{region}.csv"
        )
        updated_df.to_csv(updated_path, index=False)
        output_paths["updated_data"] = updated_path

    return output_paths


if __name__ == "__main__":
    # Example usage
    import os

    # Test with US data
    base_path = "0_prompt_llms/result/trial_1_reason_url"
    output_base_path = "1_data_analysis/results"

    for region in ["US", "UK", "Europe"]:
        print(f"\n{'='*60}")
        print(f"Processing {region}")
        print("=" * 60)

        # Process web_search_sources
        web_search_path = os.path.join(base_path, region, "web_search_sources.csv")
        if os.path.exists(web_search_path):
            print(f"\nWeb Search Sources:")
            group_counts, total_counts, df_with_common_domain = process_csv_file(
                web_search_path, region
            )
            print(f"Total unique domains: {len(total_counts)}")
            print(f"\nTop 10 domains:")
            print(total_counts.head(10))

            # Save web search sources results
            saved_paths = save_domain_counts(
                group_counts,
                total_counts,
                region,
                output_base_path,
                file_prefix="search",
                updated_df=df_with_common_domain,
            )
            print(f"\n✓ Saved web search analysis to:")
            for output_type, path in saved_paths.items():
                print(f"  - {output_type}: {path}")

        # Process response_citations
        citations_path = os.path.join(base_path, region, "response_citations.csv")
        if os.path.exists(citations_path):
            print(f"\nResponse Citations:")
            group_counts, total_counts, df_with_common_domain = process_csv_file(
                citations_path, region
            )
            print(f"Total unique domains: {len(total_counts)}")
            print(f"\nTop 10 domains:")
            print(total_counts.head(10))

            # Save response citations results
            saved_paths = save_domain_counts(
                group_counts,
                total_counts,
                region,
                output_base_path,
                file_prefix="citations",
                updated_df=df_with_common_domain,
            )
            print(f"\n✓ Saved citations analysis to:")
            for output_type, path in saved_paths.items():
                print(f"  - {output_type}: {path}")
