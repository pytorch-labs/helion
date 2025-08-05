#!/usr/bin/env python3
"""
Script to sort CSV columns alphanumerically using pandas.
Processes all CSV files in given folders and saves them with '_new' suffix.
"""

import pandas as pd
import sys
from pathlib import Path


def sort_csv_columns_pandas(input_file, output_file):
    """
    Read a CSV file and output a new CSV with columns sorted alphanumerically.
    The first column is kept in its original position.
    
    Args:
        input_file: Path to the input CSV file
        output_file: Path to the output CSV file
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Read the CSV file
        df = pd.read_csv(input_file, delimiter=';')
        
        if df.empty:
            print(f"  Warning: Empty CSV file: {input_file}")
            return False
        
        # Get column names
        columns = df.columns.tolist()
        
        if len(columns) <= 1:
            print(f"  Warning: Only one column found in {input_file}, nothing to sort")
            df.to_csv(output_file, sep=';', index=False)
            return True
        
        # Keep first column, sort the rest
        first_col = columns[0]
        other_cols = columns[1:]
        
        # Sort other columns alphanumerically
        sorted_cols = sorted(other_cols)
        
        # Create new column order
        new_column_order = [first_col] + sorted_cols
        
        # Reorder dataframe columns
        df_sorted = df[new_column_order]
        
        # Save to output file
        df_sorted.to_csv(output_file, sep=';', index=False)
        
        return True
        
    except Exception as e:
        print(f"  Error processing {input_file}: {e}")
        return False


def process_folders(folder_list):
    """
    Process all CSV files in the given folders.
    
    Args:
        folder_list: List of folder paths to process
    """
    total_processed = 0
    total_errors = 0
    
    for folder_path in folder_list:
        folder = Path(folder_path)
        
        if not folder.exists():
            print(f"\nWarning: Folder '{folder}' does not exist, skipping...")
            continue
            
        if not folder.is_dir():
            print(f"\nWarning: '{folder}' is not a directory, skipping...")
            continue
        
        print(f"\nProcessing folder: {folder}")
        print("-" * 50)
        
        # Find all CSV files in the folder (non-recursive)
        csv_files = [f for f in folder.glob("*.csv") if f.is_file()]
        
        # Filter out files that already have '_new' suffix
        csv_files = [f for f in csv_files if '_new.csv' not in f.name]
        
        if not csv_files:
            print(f"  No CSV files to process in {folder}")
            continue
        
        print(f"  Found {len(csv_files)} CSV file(s) to process")
        
        for csv_file in csv_files:
            # Create output filename with '_new' suffix
            output_file = csv_file.parent / (csv_file.stem + '_new.csv')
            
            print(f"\n  Processing: {csv_file.name}")
            print(f"  Output: {output_file.name}")
            
            if sort_csv_columns_pandas(csv_file, output_file):
                print(f"  ✓ Successfully sorted columns")
                total_processed += 1
            else:
                total_errors += 1
    
    print("\n" + "=" * 50)
    print(f"Summary: Processed {total_processed} files successfully, {total_errors} errors")


def main():
    """Main function to handle command line arguments."""
    if len(sys.argv) < 2:
        print("Usage: python sort_csv_columns_pandas.py <folder1> [folder2] [folder3] ...")
        print("Example: python sort_csv_columns_pandas.py ./data ./results ./benchmarks")
        print("\nThis script will:")
        print("  1. Find all CSV files in each specified folder")
        print("  2. Sort columns alphabetically (keeping first column fixed)")
        print("  3. Save sorted files with '_new' suffix")
        print("\nNote: Requires pandas to be installed (pip install pandas)")
        sys.exit(1)
    
    # Get list of folders from command line arguments
    folders = sys.argv[1:]
    
    print(f"CSV Column Sorter (Pandas Version)")
    print(f"Processing {len(folders)} folder(s)...")
    
    process_folders(folders)


if __name__ == "__main__":
    main()
