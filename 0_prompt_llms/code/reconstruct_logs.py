#!/usr/bin/env python3
"""
Reconstruct consolidated logs from raw response files

This script reads raw JSON response files and regenerates:
- gpt_response_metadata.json
- gpt_reasoning.json
- gpt_reasoning.md
"""

import os
import json
import glob
import re
import time
from dataclasses import dataclass
import sys

sys.path.append(os.path.join("llm_bias_analysis", "code"))
from openai_query import GPTResponse_metadata
from response_handler import response_utils, reasoning_and_web_search_response


def extract_reasoning_and_web_search(response_dict: dict) -> list[dict]:
    """
    Extract reasoning and web search from response dictionary.
    
    Includes validation to handle unexpected response structures safely.
    
    Args:
        response_dict: Dictionary representation of OpenAI API response
        
    Returns:
        List of dicts containing reasoning, web_search, and message data
    """
    result = []
    for i in response_dict["output"]:
        if i["type"] == "reasoning":
            # Validate summary structure before accessing
            if "summary" in i and isinstance(i["summary"], list):
                if len(i["summary"]) == 1:
                    # Single summary item - try to extract text
                    summary_item = i["summary"][0]
                    if isinstance(summary_item, dict) and "text" in summary_item:
                        result.append({"reasoning": summary_item["text"]})
                    else:
                        # Fallback: use the whole summary item
                        result.append({"reasoning": i["summary"]})
                else:
                    # Multiple summary items or empty list
                    result.append({"reasoning": i["summary"]})
            else:
                # No summary or invalid format
                result.append({"reasoning": "N/A"})
                
        elif i["type"] == "web_search_call":
            # Check if the web_search_call has an action field
            if "action" in i and "query" in i["action"]:
                sources = i["action"].get("sources", [])
                result.append({
                    "web_search": {
                        "url_num": len(sources),
                        "query": i["action"]["query"],
                        "sources": sources
                    }
                })
            else:
                # Handle web_search_call without action
                result.append({
                    "web_search": {
                        "url_num": 0,
                        "query": "N/A",
                        "sources": []
                    }
                })
                
        elif i["type"] == "message":
            # Validate content structure before accessing
            if "content" in i and isinstance(i["content"], list) and len(i["content"]) > 0:
                content_item = i["content"][0]
                if isinstance(content_item, dict):
                    annotations = content_item.get("annotations", [])
                    text = content_item.get("text", "")
                    result.append({
                        "message": {
                            "url_num": len(annotations),
                            "url_annotations": annotations,
                            "content": text
                        }
                    })
                else:
                    # Invalid content item format
                    result.append({
                        "message": {
                            "url_num": 0,
                            "url_annotations": [],
                            "content": str(content_item)
                        }
                    })
            else:
                # No content or empty content list
                result.append({
                    "message": {
                        "url_num": 0,
                        "url_annotations": [],
                        "content": "N/A"
                    }
                })
    return result


def count_reasoning_num(response_dict: dict) -> int:
    """Count reasoning blocks in response"""
    all_types = [t["type"] for t in response_dict["output"]]
    return all_types.count("reasoning")


def count_web_search_num(response_dict: dict) -> int:
    """Count web search calls in response"""
    all_types = [t["type"] for t in response_dict["output"]]
    return all_types.count("web_search_call")


def extract_web_search_options(response_dict: dict) -> dict | None:
    """
    Extract web search options from the tools field in the response.
    The actual location used is stored in the tools array.
    """
    tools = response_dict.get("tools", [])

    for tool in tools:
        if tool.get("type") == "web_search":
            web_search_options = {}

            # Extract search_context_size
            if "search_context_size" in tool:
                web_search_options["search_context_size"] = tool["search_context_size"]

            # Extract user_location
            if "user_location" in tool:
                user_loc = tool["user_location"]
                # Build the approximate location object
                approximate = {}

                if user_loc.get("city"):
                    approximate["city"] = user_loc["city"]
                if user_loc.get("country"):
                    approximate["country"] = user_loc["country"]
                if user_loc.get("region"):
                    approximate["region"] = user_loc["region"]
                if user_loc.get("timezone"):
                    approximate["timezone"] = user_loc["timezone"]

                if approximate:
                    web_search_options["user_location"] = {"approximate": approximate}

            return web_search_options if web_search_options else None

    return None


def extract_prompt_from_response(response_dict: dict) -> str:
    """
    Extract the original prompt from the response dictionary.
    
    The prompt is stored in response_dict["input"][0]["content"] for the user role.
    Falls back to "Unknown prompt" if not found.
    
    Args:
        response_dict: Dictionary representation of OpenAI API response
        
    Returns:
        The prompt string or "Unknown prompt" if not found
    """
    try:
        # The prompt is in the input field, typically as first message with role "user"
        if "input" in response_dict and isinstance(response_dict["input"], list):
            for msg in response_dict["input"]:
                if isinstance(msg, dict) and msg.get("role") == "user":
                    content = msg.get("content", "")
                    if content:
                        return content
        
        # Fallback: couldn't find prompt
        return "Unknown prompt"
        
    except Exception as e:
        print(f"Warning: Could not extract prompt from response: {e}")
        return "Unknown prompt"


def parse_filename(filename: str) -> dict[str, any]:
    """
    Parse filename to extract prompt index and repetition number.
    Expected format: prompt{idx}_repetition{rep}_{reasoning}.json
    Also handles: prompt{idx}_trial{rep}_{reasoning}.json
    """
    # Remove .json extension
    basename = os.path.basename(filename).replace(".json", "")

    # Try to match pattern: prompt{idx}_repetition{rep}_{reasoning}
    match = re.match(r"prompt(\d+)_(?:repetition|trial)(\d+)_(.+)", basename)
    if match:
        return {
            "prompt_idx": int(match.group(1)),
            "repetition": int(match.group(2)),
            "reasoning_effort": match.group(3),
            "basename": basename
        }

    # Fallback: just use the filename as-is
    return {
        "prompt_idx": None,
        "repetition": None,
        "reasoning_effort": None,
        "basename": basename
    }


def reconstruct_logs_from_raw_files(
    result_dir: str = None,
    file_pattern: str = "prompt*_repetition*_*.json"
) -> None:
    """
    Reconstruct consolidated logs from raw response files.

    Args:
        result_dir: Directory containing raw response files (default: llm_bias_analysis/result)
        file_pattern: Glob pattern to match raw response files
    """

    if result_dir is None:
        result_dir = os.path.join(os.path.dirname(__file__), "..", "result")

    # Find all matching files
    search_pattern = os.path.join(result_dir, file_pattern)
    raw_files = glob.glob(search_pattern)

    # Also try trial pattern
    trial_pattern = os.path.join(result_dir, "prompt*_trial*_*.json")
    raw_files.extend(glob.glob(trial_pattern))

    # Remove duplicates
    raw_files = list(set(raw_files))

    if not raw_files:
        print(f"No raw response files found matching pattern: {file_pattern}")
        print(f"Searched in: {result_dir}")
        return

    print(f"\n{'='*80}")
    print(f"Found {len(raw_files)} raw response files")
    print(f"{'='*80}\n")

    # Sort files by prompt index and repetition
    def sort_key(filepath):
        info = parse_filename(filepath)
        prompt_idx = info["prompt_idx"] if info["prompt_idx"] is not None else 999
        repetition = info["repetition"] if info["repetition"] is not None else 999
        return (prompt_idx, repetition)

    raw_files.sort(key=sort_key)

    responses_log = []
    reasoning_log = []

    # Process each file
    for filepath in raw_files:
        filename_info = parse_filename(filepath)
        basename = filename_info["basename"]

        print(f"Processing: {os.path.basename(filepath)}")

        try:
            # Load raw response
            with open(filepath, 'r', encoding='utf-8') as f:
                response_dict = json.load(f)

            # Extract information - get prompt from response data
            prompt = extract_prompt_from_response(response_dict)
            output_text = ""

            # Extract output text from message blocks
            for output_block in response_dict.get("output", []):
                if output_block.get("type") == "message":
                    if "content" in output_block and len(output_block["content"]) > 0:
                        output_text += output_block["content"][0].get("text", "")

            detailed_response = extract_reasoning_and_web_search(response_dict)
            source_num = sum([
                i.get("web_search", {}).get("url_num", 0)
                for i in detailed_response
                if "web_search" in i.keys()
            ])

            # Extract web search options from tools field (actual location used)
            web_search_options = extract_web_search_options(response_dict)

            # Create metadata object
            metadata = GPTResponse_metadata(
                filename=basename,
                prompt=prompt,
                response=output_text,
                reasoning_num=count_reasoning_num(response_dict),
                web_search_request_num=count_web_search_num(response_dict),
                web_source_num=source_num,
                model=response_dict.get("model", "unknown"),
                timestamp=response_dict.get("created_at", time.strftime("%Y-%m-%d %H:%M:%S")),
                temperature=response_dict.get("temperature", 1.0),
                max_output_tokens=response_dict.get("max_output_tokens", 0),
                reasoning_effort=response_dict.get("reasoning", {}).get("effort", "unknown"),
                verbosity=response_dict.get("text", {}).get("verbosity", "unknown"),
                store=response_dict.get("store", False),
                safety_identifier=response_dict.get("safety_identifier"),
                tools=response_dict.get("tools"),
                include=response_dict.get("include"),
                prompt_tokens=response_dict.get("usage", {}).get("input_tokens"),
                reasoning_tokens=response_dict.get("usage", {}).get("output_tokens_details", {}).get("reasoning_tokens"),
                total_tokens=response_dict.get("usage", {}).get("total_tokens"),
                id=response_dict.get("id", "unknown"),
                web_search_options=web_search_options
            )

            responses_log.append(metadata)

            # Create reasoning object
            reasoning = reasoning_and_web_search_response(
                filename=basename,
                prompt=prompt,
                detailed_response=detailed_response,
                id=response_dict.get("id", "unknown"),
                web_search_options=web_search_options
            )

            reasoning_log.append(reasoning)

            print(f"  ✓ Processed successfully")

        except Exception as e:
            print(f"  ✗ Error processing {os.path.basename(filepath)}: {str(e)}")
            continue

    # Save consolidated logs
    if responses_log:
        print(f"\n{'='*80}")
        print(f"Saving consolidated logs (OVERWRITE MODE)...")
        print(f"{'='*80}\n")

        response_handler = response_utils(result_dir=result_dir)

        # Delete existing files to ensure overwrite
        metadata_path = os.path.join(result_dir, "gpt_response_metadata.json")
        reasoning_json_path = os.path.join(result_dir, "gpt_reasoning.json")
        reasoning_md_path = os.path.join(result_dir, "gpt_reasoning.md")

        for path in [metadata_path, reasoning_json_path, reasoning_md_path]:
            if os.path.exists(path):
                os.remove(path)
                print(f"Removed existing file: {os.path.basename(path)}")

        # Save metadata log
        response_handler.save_responses_log(
            responses_log,
            "gpt_response_metadata"
        )

        # Save reasoning and web search log (JSON)
        response_handler.save_responses_log(
            reasoning_log,
            "gpt_reasoning"
        )

        # Save reasoning and web search log (Markdown)
        response_handler.save_reasoning_markdown(
            reasoning_log,
            "gpt_reasoning"
        )

        print(f"\n{'='*80}")
        print(f"Reconstruction complete!")
        print(f"  - Processed files: {len(responses_log)}/{len(raw_files)}")
        print(f"  - Logs saved to: {result_dir}")
        print(f"{'='*80}\n")
    else:
        print("\n⚠ No responses were successfully processed")


def main():
    """
    Main function to reconstruct logs from raw files
    """
    print("\n" + "="*80)
    print("RECONSTRUCT CONSOLIDATED LOGS FROM RAW FILES")
    print("="*80)

    # Ask user for custom pattern or use default
    print("\nDefault pattern: prompt*_repetition*_*.json")
    custom_pattern = input("Enter custom pattern (or press Enter for default): ").strip()

    if not custom_pattern:
        custom_pattern = "prompt*_repetition*_*.json"  # More flexible pattern

    print(f"\nUsing pattern: {custom_pattern}")

    # Confirm
    confirm = input("\nProceed with reconstruction? (yes/no): ").strip().lower()
    if confirm not in ['yes', 'y']:
        print("Cancelled.")
        return

    try:
        reconstruct_logs_from_raw_files(file_pattern=custom_pattern)
    except Exception as e:
        print(f"\nError during reconstruction: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()