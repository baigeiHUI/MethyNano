import pandas as pd
from tqdm import tqdm
import argparse
import os
import tempfile

def process_tsv_file(coord_dict, tsv_path, output_file, chunksize=200000, first_chunk_global=True):

    first_chunk = first_chunk_global
    for chunk in tqdm(
        pd.read_csv(tsv_path, sep='\t', chunksize=chunksize, header=None),
        desc=f"Processing {os.path.basename(tsv_path)}"
    ):
        base_cols = [
            'id','chr','start','sequence',
            'current_mean','current_var','current_median',
            'current_length','base_quality'
        ]
        sig_n = max(0, chunk.shape[1] - len(base_cols))
        chunk.columns = base_cols + [f'current_signal{i}' for i in range(1, sig_n + 1)]
        
        chunk_clean = chunk.drop(columns=['id'])
        chunk_clean['chr'] = 'chr' + chunk_clean['chr'].astype(str)
        chunk_clean['c_pos'] = chunk_clean['start'] + (sig_n // 2)
        chunk_clean['coord_key'] = chunk_clean['chr'] + ':' + chunk_clean['c_pos'].astype(str)
        
        chunk_clean['methylation'] = chunk_clean['coord_key'].map(coord_dict.get)
        
        aligned_chunk = chunk_clean.dropna(subset=['methylation']).drop(columns=['coord_key', 'c_pos'])
        aligned_chunk.to_csv(
            output_file,
            sep='\t',
            mode='a' if not first_chunk else 'w',
            header=first_chunk,
            index=False,
            float_format='%.4f'
        )
        first_chunk = False
    return first_chunk

def extract_unique_coords(tsv_paths, chunksize):

    print(f"\n{'='*40}\n[Memory Optimization] Pass 1: Scanning TSV files to extract unique coordinates...")
    unique_coords = set()
    for tsv_path in tsv_paths:
        for chunk in tqdm(
            pd.read_csv(tsv_path, sep='\t', chunksize=chunksize, header=None),
            desc=f"Scanning {os.path.basename(tsv_path)}"
        ):
            sig_n = max(0, chunk.shape[1] - 9)
            chr_col = 'chr' + chunk[1].astype(str)
            pos_col = chunk[2] + (sig_n // 2)
            coords = chr_col + ':' + pos_col.astype(str)
            unique_coords.update(coords.tolist())
    print(f"Found {len(unique_coords)} unique coordinates.")
    return unique_coords

def load_minimal_coord_dict(bed_path, coords_to_find):
  
    print(f"\n{'='*40}\n[Memory Optimization] Pass 2: Loading methylation data from BED file...")
    coords_to_find = set(coords_to_find)
    minimal_coord_dict = {}
    bed_dtypes = {'chr': 'category', 'start': 'int32', 'end': 'int32', 'RGB': 'str', 'coverage': 'int32', 'methylation': 'int32'}
    bed_chunksize = 1_000_000
    found_count = 0

    for bed_chunk in tqdm(
        pd.read_csv(bed_path, sep='\s+', header=0, dtype=bed_dtypes, chunksize=bed_chunksize),
        desc="Loading relevant BED data"
    ):
        bed_chunk['chr'] = bed_chunk['chr'].astype(str)
        bed_chunk = bed_chunk.drop_duplicates(subset=['chr', 'start'], keep='first')
        bed_chunk_keys = bed_chunk['chr'] + ':' + bed_chunk['start'].astype(str)
        mask = bed_chunk_keys.isin(coords_to_find)
        relevant_chunk = bed_chunk[mask]
        
        if not relevant_chunk.empty:
            minimal_coord_dict.update(zip(relevant_chunk['chr'] + ':' + relevant_chunk['start'].astype(str), relevant_chunk['methylation']))
            found_count += len(relevant_chunk)
            if found_count == len(coords_to_find):
                print("All required methylation data has been found. Ending loading early.")
                break
    print(f"Successfully loaded {len(minimal_coord_dict)} methylation records.")
    return minimal_coord_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Align methylation data (Memory Optimized)')
    parser.add_argument('--bed', required=True, help='Path to methylation BED file')
    parser.add_argument('--tsv', required=True, nargs='+', help='Input TSV file paths (wildcards supported)')
    parser.add_argument('--output_file', required=True, help='Output file path')
    parser.add_argument('--chunksize', type=int, default=500000, help='Processing chunk size (optimized for memory)')
    
    args = parser.parse_args()
    

    coords_to_load = extract_unique_coords(args.tsv, args.chunksize)
 
    coord_dict = load_minimal_coord_dict(args.bed, coords_to_load)

    print(f"\n{'='*40}\nStarting to process sequence feature data and write final results...")
    first_chunk_global = True
    for tsv_path in args.tsv:
        print(f"\nProcessing file: {os.path.basename(tsv_path)}")
        first_chunk_global = process_tsv_file(
            coord_dict=coord_dict,
            tsv_path=tsv_path,
            output_file=args.output_file,
            chunksize=args.chunksize,
            first_chunk_global=first_chunk_global
        )
    
    print(f"\n{'='*40}\nProcessing completed! Results saved to: {os.path.abspath(args.output_file)}")
