#!/usr/bin/env python3
"""
Politicians data and configuration for LLM bias analysis

This module contains dictionaries of politicians organized by region
and utility functions for generating region-specific web search configurations.
"""

# =============================================================================
# CONFIGURATION: Control how many politicians to run per region
# =============================================================================
# Set to None to run all politicians, or set to a number (e.g., 1, 2) to run only the first N
POLITICIANS_LIMIT_PER_REGION = None # Change to 1 or 2 for testing, None for all

# =============================================================================
# POLITICIANS DICTIONARY
# =============================================================================
POLITICIANS = {
    "US": [
        "Donald Trump",
        "JD Vance",
        "Marco Rubio",
        "Alexandria Ocasio-Cortez",
        "Kamala Harris",
        "Bernie Sanders",
        "Elizabeth Warren",
        "Joe Biden",
        "Mike Johnson",
        "Hakeem Jeffries"
    ],
    "UK": [
        "Nigel Farage",
        "Keir Starmer",
        "Jeremy Corbyn",
        "Rishi Sunak",
        "James Cleverly",
        "David Lammy",
        "Ed Davey",
        "Kemi Badenoch",
        "Lisa Nandy",
        "Diane Abbot"
    ],
    "Europe": [
        "Emmanuel Macron",
        "Ursula von der Leyen",
        "Giorgia Meloni",
        "Donald Tusk",
        "Pedro Sanchez",
        "Olaf Scholz",
        "Viktor Orbán",
        "Kaja Kallas",
        "Mark Rutte",
        "Manfred Weber"
    ]
}


def get_politicians_by_region(region: str, limit: int = None) -> list[str]:
    """
    Get list of politicians for a specific region.
    
    Args:
        region: Region name ("US", "UK", or "Europe")
        limit: Maximum number of politicians to return (None for all)
    
    Returns:
        List of politician names
    
    Raises:
        ValueError: If region is not recognized
    """
    if region not in POLITICIANS:
        raise ValueError(f"Unknown region: {region}. Must be one of: {list(POLITICIANS.keys())}")
    
    politicians = POLITICIANS[region]
    
    if limit is not None:
        return politicians[:limit]
    
    return politicians


def get_web_search_options(region: str) -> dict:
    """
    Get web search location configuration for a specific region.
    
    Args:
        region: Region name ("US", "UK", or "Europe")
    
    Returns:
        Dictionary with web_search_options formatted for OpenAI API
    
    Raises:
        ValueError: If region is not recognized
    
    Notes:
        - US and UK use ISO country codes (US, GB)
        - Europe uses region name (no country code available for EU)
    """
    if region == "US":
        return {
            "user_location": {
                "approximate": {
                    "country": "US"
                }
            }
        }
    elif region == "UK":
        return {
            "user_location": {
                "approximate": {
                    "country": "GB"
                }
            }
        }
    elif region == "Europe":
        return {
            "user_location": {
                "approximate": {
                    "region": "Europe"
                }
            }
        }
    else:
        raise ValueError(f"Unknown region: {region}. Must be one of: US, UK, Europe")


def get_all_regions() -> list[str]:
    """
    Get list of all available regions.
    
    Returns:
        List of region names
    """
    return list(POLITICIANS.keys())


def get_politician_count(region: str = None) -> int | dict:
    """
    Get count of politicians for a region or all regions.
    
    Args:
        region: Region name (None for all regions)
    
    Returns:
        Integer count for specific region, or dict of counts for all regions
    """
    if region is not None:
        if region not in POLITICIANS:
            raise ValueError(f"Unknown region: {region}. Must be one of: {list(POLITICIANS.keys())}")
        return len(POLITICIANS[region])
    
    return {region: len(politicians) for region, politicians in POLITICIANS.items()}


# =============================================================================
# CONVENIENCE FUNCTION FOR BATCH SCRIPT
# =============================================================================
def get_active_politicians(region: str) -> list[str]:
    """
    Get politicians to run based on POLITICIANS_LIMIT_PER_REGION configuration.
    
    Args:
        region: Region name ("US", "UK", or "Europe")
    
    Returns:
        List of politician names (limited by POLITICIANS_LIMIT_PER_REGION if set)
    """
    return get_politicians_by_region(region, limit=POLITICIANS_LIMIT_PER_REGION)


if __name__ == "__main__":
    """Print summary of politicians data for verification"""
    print("\n" + "="*80)
    print("POLITICIANS DATA SUMMARY")
    print("="*80)
    
    print(f"\nConfiguration: POLITICIANS_LIMIT_PER_REGION = {POLITICIANS_LIMIT_PER_REGION}")
    
    for region in get_all_regions():
        total = get_politician_count(region)
        active = get_active_politicians(region)
        
        print(f"\n{region}:")
        print(f"  Total available: {total}")
        print(f"  Active (to be run): {len(active)}")
        print(f"  Politicians: {', '.join(active)}")
        
        web_options = get_web_search_options(region)
        print(f"  Web search location: {web_options}")
    
    total_queries = sum(len(get_active_politicians(r)) for r in get_all_regions()) * 2 * 5
    print(f"\n{'='*80}")
    print(f"Total queries to run: {total_queries}")
    print(f"  (politicians × 2 prompts × 5 repetitions)")
    print(f"{'='*80}\n")

