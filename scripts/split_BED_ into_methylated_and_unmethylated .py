import pandas as pd
import argparse
from tqdm import tqdm
import os

def split_bed_by_methylation(bed_path, output_prefix, chunksize=1000000):
    """
    Efficiently split a BED file into methylated (1) and unmethylated (0) files.
    
    Args:
        bed_path (str): Path to the input BED file.
        output_prefix (str): Prefix for output files (e.g., "cpg").
        chunksize (int): Number of rows per processing chunk (default: 1,000,000).
    """
    # Define output filenames
    output_1 = f"{output_prefix}_1.bed"
    output_0 = f"{output_prefix}_0.bed"
    
    # BED6 column names and data types
    bed_columns = ['chr', 'start', 'end', 'RGB', 'coverage', 'methylation']
    bed_dtypes = {
        'chr': 'str',
        'start': 'int32',
        'end': 'int32',
        'RGB': 'str',
        'coverage': 'int32',
        'methylation': 'int8'  # Memory efficient
    }
    
    # Flags to control header writing
    first_chunk_1 = True
    first_chunk_0 = True
    
    # Statistics counters
    total_processed = 0
    total_methylated = 0
    total_unmethylated = 0
    
    try:
        for chunk in tqdm(
            pd.read_csv(
                bed_path,
                sep=r'\s+',
                header=None,
                names=bed_columns,
                dtype=bed_dtypes,
                chunksize=chunksize
            ),
            desc="Processing BED file"
        ):
            total_processed += len(chunk)
            
            # Split into methylated and unmethylated
            methylated = chunk[chunk['methylation'] == 1]
            unmethylated = chunk[chunk['methylation'] == 0]
            
            total_methylated += len(methylated)
            total_unmethylated += len(unmethylated)
            
            # Write methylated records
            if not methylated.empty:
                methylated.to_csv(
                    output_1,
                    sep='\t',
                    mode='a' if not first_chunk_1 else 'w',
                    header=first_chunk_1,
                    index=False
                )
                first_chunk_1 = False
            
            # Write unmethylated records
            if not unmethylated.empty:
                unmethylated.to_csv(
                    output_0,
                    sep='\t',
                    mode='a' if not first_chunk_0 else 'w',
                    header=first_chunk_0,
                    index=False
                )
                first_chunk_0 = False
        
        # Print final statistics
        print(f"\n{'='*50}")
        print(f"Processing completed!")
        print(f"Total records: {total_processed:,}")
        print(f"Methylated (1): {total_methylated:,} → {output_1}")
        print(f"Unmethylated (0): {total_unmethylated:,} → {output_0}")
        print(f"Other values: {total_processed - total_methylated - total_unmethylated:,}")
        print(f"{'='*50}")
        
    except KeyboardInterrupt:
        print("\n  Process interrupted by user. Partial results have been saved.")
        raise
    except Exception as e:
        print(f" Error during processing: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Efficiently split a BED file into methylated (1) and unmethylated (0) files.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--bed', required=True, 
                      help='Path to the input BED file')
    parser.add_argument('--output_prefix', default='methylation',
                      help='Prefix for output files')
    parser.add_argument('--chunksize', type=int, default=1000000,
                      help='Chunk size (adjust based on available memory)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.bed):
        print(f"File not found: {args.bed}")
        exit(1)
    
    print(f"Starting processing: {args.bed}")
    print(f"Output files: {args.output_prefix}_1.bed, {args.output_prefix}_0.bed")
    
    split_bed_by_methylation(
        bed_path=args.bed,
        output_prefix=args.output_prefix,
        chunksize=args.chunksize
    )