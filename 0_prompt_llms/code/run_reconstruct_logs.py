#!/usr/bin/env python3
"""
Run log reconstruction from raw response files
"""

import os
import sys

# Add current directory to path for imports
sys.path.append(os.path.join("llm_bias_analysis", "code"))
from reconstruct_logs import reconstruct_logs_from_raw_files


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
        custom_pattern = "prompt*_*.json"  # More flexible pattern

    print(f"\nUsing pattern: {custom_pattern}")

    # Ask for custom result directory
    print("\nDefault directory: llm_bias_analysis/result")
    custom_dir = input("Enter custom directory path (or press Enter for default): ").strip()

    result_dir = custom_dir if custom_dir else None

    # Confirm
    confirm = input("\nProceed with reconstruction? (yes/no): ").strip().lower()
    if confirm not in ['yes', 'y']:
        print("Cancelled.")
        return

    try:
        reconstruct_logs_from_raw_files(result_dir=result_dir, file_pattern=custom_pattern)
    except Exception as e:
        print(f"\nError during reconstruction: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

