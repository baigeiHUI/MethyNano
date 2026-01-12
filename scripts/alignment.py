# alignment_memory_optimized.py
import pandas as pd
from tqdm import tqdm
import argparse
import os
import tempfile

def process_tsv_file(coord_dict, tsv_path, output_file, chunksize=200000, first_chunk_global=True):
    """ 处理单个TSV文件 (此函数无需修改) """
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
    """(第一遍) 从所有TSV文件中提取不重复的坐标"""
    print(f"\n{'='*40}\n[内存优化] 第一遍: 扫描TSV文件，提取唯一坐标...")
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
    print(f"共找到 {len(unique_coords)} 个唯一坐标。")
    return unique_coords

def load_minimal_coord_dict(bed_path, coords_to_find):
    """(第二遍) 根据坐标列表，从BED文件加载一个最小的字典"""
    print(f"\n{'='*40}\n[内存优化] 第二遍: 从BED文件加载甲基化数据...")
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
                print("所有需要的甲基化数据均已找到，提前结束加载。")
                break
    print(f"成功加载了 {len(minimal_coord_dict)} 条甲基化数据。")
    return minimal_coord_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Align methylation data (Memory Optimized)')
    parser.add_argument('--bed', required=True, help='甲基化BED文件路径')
    parser.add_argument('--tsv', required=True, nargs='+', help='输入TSV文件路径（支持通配符）')
    parser.add_argument('--output_file', required=True, help='输出文件路径')
    parser.add_argument('--chunksize', type=int, default=200000, help='处理块大小（已优化为内存友好）')
    
    args = parser.parse_args()
    
    # --- 优化后的主流程 ---
    # 1. 扫描TSV，获取所有唯一坐标
    coords_to_load = extract_unique_coords(args.tsv, args.chunksize)
    
    # 2. 从BED文件加载一个“迷你字典”
    coord_dict = load_minimal_coord_dict(args.bed, coords_to_load)
    
    # 3. 再次遍历TSV文件，进行匹配和输出
    print(f"\n{'='*40}\n开始处理序列特征数据并写入最终结果...")
    first_chunk_global = True
    for tsv_path in args.tsv:
        print(f"\n处理文件: {os.path.basename(tsv_path)}")
        first_chunk_global = process_tsv_file(
            coord_dict=coord_dict,
            tsv_path=tsv_path,
            output_file=args.output_file,
            chunksize=args.chunksize,
            first_chunk_global=first_chunk_global
        )
    
    print(f"\n{'='*40}\n处理完成！结果保存在: {os.path.abspath(args.output_file)}")