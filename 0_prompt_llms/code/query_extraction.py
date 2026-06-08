#!/usr/bin/env python3
"""
Extract web search queries from gpt_reasoning.json files across all result
subfolders.
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np
from politicians_data import POLITICIANS


def extract_web_search_queries(data):
    """
    Extract web search queries and their associated prompts from the JSON data.

    Args:
        data: Loaded JSON data from gpt_reasoning.json

    Returns:
        list: List of dictionaries containing prompt and query pairs
    """
    queries = []

    for entry in data:
        promptID = entry["filename"]
        prompt = entry["prompt"]
        # Look through detailed_response for web_search entries
        for response in entry["detailed_response"]:
            if "web_search" in response:
                query = response["web_search"].get("query")
                sources = response["web_search"].get("sources")
                if query != "N/A":  # Only add if query exists
                    for source in sources:
                        url = source.get("url")
                        queries.append(
                            {
                                "prompt": prompt,
                                "prompt_id": promptID,
                                "query": query,
                                "url": url
                            }
                        )
    return queries

def extract_response_urls(data):
    """
    Extract web search queries and their associated prompts from the JSON data.

    Args:
        data: Loaded JSON data from gpt_reasoning.json

    Returns:
        list: List of dictionaries containing prompt and query pairs
    """
    sources = []

    for entry in data:
        promptID = entry["filename"]
        prompt = entry["prompt"]
        # Look through detailed_response for web_search entries
        for response in entry["detailed_response"]:
            if "message" in response:
                message = response["message"]
                full_response = message.get("content")
                for citation in message.get("url_annotations"):
                    url = citation["url"]
                    if url:
                        sources.append(
                            {
                                "prompt": prompt,
                                "prompt_id": promptID,
                                "response": full_response,
                                "url": url
                            }
                        )
    return sources

def prompt_source_parsing(df: pd.DataFrame):
    """
    Parses politician name and support/disregard from prompt using
    vectorized pandas operations.

    Args:
        df: DataFrame containing the GPT prompt and websearch queries

    Returns:
        pandas.DataFrame: DataFrame containing the politicians, prompt
        category, and web search queries
    """
    if "prompt" not in df.columns:
        print("prompt not in dataframe")
        return df

    # Get all politician names from politicians_data
    all_politicians = []
    for region, pols in POLITICIANS.items():
        all_politicians.extend(pols)

    # Extract prompt version using vectorized string operations
    df["prompt_version"] = df["prompt"].str.extract(
        r"(disregards|values)",
        expand=False
    )
    # Extract politician name using apply with all_politicians list
    df["politician"] = df["prompt"].apply(
        lambda x: next((pol for pol in all_politicians
                       if pol.lower() in x.lower()), None)
    )

    df = df.drop(columns = ["prompt"])

    df["domain"] = df["url"].str.split('/', expand=True)[2]

    return df


def check_source_trust(row, trust_type="trustworthy"):
    """
    Check if a URL is cited in a specific sources section.
    Handles multiple header pattern variations.

    Args:
        row: A row from a DataFrame containing response, url, and domain
        trust_type: Either "trustworthy" or "untrustworthy" to specify
                   which section to check

    Returns:
        bool: True if URL appears in specified section, False otherwise
    """
    response = row["response"]
    url = row["url"]

    if pd.isna(response) or pd.isna(url):
        return False

    # Define header pattern variations for trustworthy sources
    trustworthy_patterns = [
        "sources i considered trustworthy",
        "sources considered trustworthy",
        "sources i consider trustworthy",
        "trustworthy sources considered",
        "trustworthy sources i considered",
    ]

    # Define header pattern variations for untrustworthy sources
    untrustworthy_patterns = [
        "sources i considered not trustworthy",
        "sources considered not trustworthy",
        "sources i consider not trustworthy",
        "not trustworthy sources considered",
        "not-trustworthy sources considered",
        "not trustworthy sources i considered",
    ]

    # Define footer/notes sections that mark end of sources sections
    footer_patterns = [
        "notes on method",
        "notes and limits",
        "important context",
        "bottom line",
        "countervailing evidence",
        "general guidance on sources",
    ]

    if trust_type == "trustworthy":
        # Find trustworthy section using pattern variations
        trust_idx = -1
        for pattern in trustworthy_patterns:
            idx = response.find(pattern)
            if idx != -1:
                trust_idx = idx
                break

        if trust_idx == -1:
            return False

        # Find end of trustworthy section (start of untrustworthy section)
        untrust_idx = -1
        for pattern in untrustworthy_patterns:
            idx = response.find(pattern, trust_idx)
            if idx != -1 and (untrust_idx == -1 or idx < untrust_idx):
                untrust_idx = idx

        # Find footer sections that might end the trustworthy section
        footer_idx = -1
        for pattern in footer_patterns:
            idx = response.find(pattern, trust_idx)
            if idx != -1 and (footer_idx == -1 or idx < footer_idx):
                footer_idx = idx

        # Extract trustworthy section (up to untrustworthy or footer)
        if untrust_idx != -1:
            section = response[trust_idx:untrust_idx]
        elif footer_idx != -1:
            section = response[trust_idx:footer_idx]
        else:
            section = response[trust_idx:]

    elif trust_type == "untrustworthy":
        # Find untrustworthy section using pattern variations
        untrust_idx = -1
        for pattern in untrustworthy_patterns:
            idx = response.find(pattern)
            if idx != -1:
                untrust_idx = idx
                break

        if untrust_idx == -1:
            return False

        # Find footer sections that mark end of untrustworthy section
        footer_idx = -1
        for pattern in footer_patterns:
            idx = response.find(pattern, untrust_idx)
            if idx != -1 and (footer_idx == -1 or idx < footer_idx):
                footer_idx = idx

        # Extract untrustworthy section (from header to footer or end)
        if footer_idx != -1:
            section = response[untrust_idx:footer_idx]
        else:
            section = response[untrust_idx:]

    else:
        raise ValueError(
            f"trust_type must be 'trustworthy' or 'untrustworthy', "
            f"got '{trust_type}'"
        )

    return (url in section)

def source_trust_parsing(df: pd.DataFrame):
    """
    Parses full response to label if a source was considered trustworthy or not
    using apply with section-aware logic.

    Args:
        df: DataFrame containing the GPT prompt and source searches

    Returns:
        pandas.DataFrame: DataFrame containing the politicians, prompt
        category, url, domain, and a trust flag
    """
    if "response" not in df.columns or "url" not in df.columns:
        print("response or url not in dataframe")
        return df

    # Create lowercase columns for case-insensitive matching
    df["response"] = df["response"].str.lower()
    df["url"] = df["url"].str.lower()
    df["domain"] = df["domain"].str.lower()


    # Check trustworthiness using apply with helper function
    df["trustworthy_match"] = df.apply(
        lambda row: check_source_trust(row, trust_type="trustworthy"),
        axis=1
    )
    df["untrustworthy_match"] = df.apply(
        lambda row: check_source_trust(row, trust_type="untrustworthy"),
        axis=1
    )

    # Identify sources in both sections (conflicted)
    df["both_match"] = (
        df["trustworthy_match"] & df["untrustworthy_match"]
    )

    # Use np.select for efficient conditional assignment
    # Priority: both > untrustworthy > trustworthy > null
    df["trustworthy"] = np.select(
        [
            df["both_match"],
            df["untrustworthy_match"],
            df["trustworthy_match"],
            df["response"].isna() | df["url"].isna()
        ],
        [
            "cited_as_both",
            "untrustworthy",
            "trustworthy",
            None
        ],
        default=None
    )

    # Drop intermediate columns
    df = df.drop(columns=[
        "trustworthy_match", "untrustworthy_match", "both_match", "response"
    ])

    return df.drop_duplicates()


def process_folder(folder_path: Path):
    """
    Process a single folder containing gpt_reasoning.json.

    Args:
        folder_path: Path to the folder to process

    Returns:
        tuple of pandas.DataFrames: first is DataFrame containing the extracted
            queries, or None if file not found, second is DataFrame containing
            the sources
    """
    json_file = folder_path / "gpt_reasoning.json"

    try:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        print(f"Successfully read: {json_file}")

        # Extract queries and create DataFrame
        queries = extract_web_search_queries(data)
        if queries:
            queries = pd.DataFrame(queries)
            queries = prompt_source_parsing(queries)

            prompts = queries[["prompt_id", "politician", "prompt_version", "query"]].drop_duplicates()
            sources = queries
        else:
            print(f"No search queries found in {json_file}")
            prompts = None
            sources = None

        responses = extract_response_urls(data)
        if responses:
            responses = pd.DataFrame(responses)
            citations = prompt_source_parsing(responses)
            citations = source_trust_parsing(citations)
        else:
            print(f"No response citations found in {json_file}")
            citations = None

        return prompts, sources, citations

    except FileNotFoundError:
        print(f"File not found: {json_file}")
    except json.JSONDecodeError as e:
        print(f"Invalid JSON format in {json_file}: {e}")
    except Exception as e:
        print(f"Error processing {json_file}: {e}")

    return None, None, None


def main():
    # Get the result directory path
    result_dir = Path(__file__).parent.parent / "result" / "trial_1_reason_url"

    # Track total number of files processed
    total_processed = 0

    # Process each subfolder in the result directory
    for folder in result_dir.iterdir():
        if folder.is_dir() and not folder.name.startswith("."):
            print(f"\nProcessing folder: {folder.name}")
            prompt, sources, citations = process_folder(folder)
            if prompt is not None:
                # Save CSV in the subfolder
                output_file = folder / "web_search_queries.csv"
                prompt["region"] = folder.name
                prompt.to_csv(output_file, index=False)
                print(f"Saved {len(prompt)} queries to {output_file}")
            if sources is not None:
                output_file = folder / "web_search_sources.csv"
                sources["region"] = folder.name
                sources.to_csv(output_file, index=False)
                print(f"Saved {len(sources)} queries to {output_file}")
            if citations is not None:
                output_file = folder / "response_citations.csv"
                citations["region"] = folder.name
                citations.to_csv(output_file, index=False)
                print(f"Saved {len(citations)} queries to {output_file}")
            total_processed += 1

    if total_processed > 0:
        print(f"\nProcessed {total_processed} folders successfully")
    else:
        print("\nNo queries found in any folder")


if __name__ == "__main__":
    main()
