import pandas as pd
import os

def split_methylation_file(input_file, chunk_size=100000, sep='\t'):
    """
    Read a large file and split it by motif type (CG, CHG, CHH).
    
    :param input_file: Path to the input file.
    :param chunk_size: Number of rows to read at a time. 100,000 rows typically uses tens of MB of memory—very safe.
    :param sep: Delimiter. Based on your data preview, this is usually '\t' (tab). Change to ',' if comma-separated.
    """
    
    print(f"Processing file: {input_file} ...")
    
    # Define output filenames
    base_name = os.path.splitext(input_file)[0]
    out_files = {
        'CG': f"{base_name}_CG.csv",
        'CHG': f"{base_name}_CHG.csv",
        'CHH': f"{base_name}_CHH.csv"
    }
    
    # Remove existing output files to avoid duplicate appending
    for f in out_files.values():
        if os.path.exists(f):
            os.remove(f)
            
    # Flag to control header writing
    first_chunk = True
    
    # Read file in chunks
    # usecols is optional; omitted here to read all columns by default.
    reader = pd.read_csv(input_file, sep=sep, chunksize=chunk_size, iterator=True)
    
    total_rows = 0
    
    for chunk in reader:
 
        next_base = chunk['sequence'].str[7]

        next_next_base = chunk['sequence'].str[8]
        
        cg_mask = (next_base == 'G')
        cg_data = chunk[cg_mask]
        

        chg_mask = (next_base != 'G') & (next_next_base == 'G')
        chg_data = chunk[chg_mask]

        chh_mask = (next_base != 'G') & (next_next_base != 'G')
        chh_data = chunk[chh_mask]
        
        # Write to output files
        mode = 'w' if first_chunk else 'a'
        header = True if first_chunk else False
        
        if not cg_data.empty:
            cg_data.to_csv(out_files['CG'], index=False, sep=sep, mode=mode, header=header)
        if not chg_data.empty:
            chg_data.to_csv(out_files['CHG'], index=False, sep=sep, mode=mode, header=header)
        if not chh_data.empty:
            chh_data.to_csv(out_files['CHH'], index=False, sep=sep, mode=mode, header=header)
            
        total_rows += len(chunk)
        print(f"Processed {total_rows} rows...", end='\r')
        
        first_chunk = False

    print(f"\nFile {input_file} processing complete!")
    print(f"Output files: \n - {out_files['CG']} \n - {out_files['CHG']} \n - {out_files['CHH']}")


# ==========================================
# Execution Section
# ==========================================

# List of files to process
files_to_process = ['all_pos.csv', 'all_neg.csv']

for file_path in files_to_process:
    if os.path.exists(file_path):
        # Note: Based on your data sample, columns appear to be separated by tabs or spaces.
        # If your file is standard comma-separated CSV, change sep='\t' to sep=','
        split_methylation_file(file_path, sep='\t') 
    else:
        print(f"Error: File not found: {file_path}")
