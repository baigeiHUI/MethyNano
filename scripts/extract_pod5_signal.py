#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract signal from POD5 files using a coordinate-sorted BAM.
Fixed: pickle error by converting AlignedSegment to dict.
"""

import argparse
import os
import sys
import glob
import multiprocessing
from functools import partial
from tqdm import tqdm
import numpy as np
import pysam
import pod5


def get_signal_segments_pA(raw_signal, move_table, stride, start_idx):
    move_indices = np.where(move_table)[0]
    segments = []
    for i in range(len(move_indices) - 1):
        seg_start = move_indices[i] * stride + start_idx
        seg_end = move_indices[i + 1] * stride + start_idx
        segments.append(raw_signal[seg_start:seg_end])
    if len(move_indices) > 0:
        avg_len = int(np.mean([len(s) for s in segments])) if segments else 5
        last_start = move_indices[-1] * stride + start_idx
        last_end = last_start + avg_len
        segments.append(raw_signal[last_start:last_end])
    return segments


def reverse_complement(seq):
    comp = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
    return ''.join(comp.get(base, 'N') for base in reversed(seq.upper()))


def process_pod5_file(pod5_path, bam_dict, reference_fasta, output_queue, clip=10):
    try:
        ref_fasta = pysam.FastaFile(reference_fasta)
        results = []

        with pod5.Reader(pod5_path) as reader:
            for read_record in reader.reads():
                read_id = str(read_record.read_id)
                if read_id not in bam_dict:
                    continue

                aln_data = bam_dict[read_id]

                ref_name = aln_data['ref_name']
                ref_start_0b = aln_data['ref_start']
                ref_end_0b = aln_data['ref_end']
                query_seq = aln_data['query_seq']
                qual = aln_data['qual']
                is_reverse = aln_data['is_reverse']
                tags = aln_data['tags']

                try:
                    ref_seq_full = ref_fasta.fetch(ref_name, ref_start_0b, ref_end_0b).upper()
                except Exception as e:
                    print(f"Failed to fetch {ref_name}:{ref_start_0b}-{ref_end_0b}: {e}", file=sys.stderr)
                    continue

                if is_reverse:
                    query_seq = reverse_complement(query_seq)
                    qual = qual[::-1]

                if 'mv' not in tags or 'ts' not in tags:
                    continue
                mv = tags['mv']
                stride = int(mv[0])
                move_table = np.array(mv[1:], dtype=np.int8)
                ts = tags['ts']

                raw_signal = read_record.signal.astype(np.float64)
                offset = read_record.calibration.offset
                scale = read_record.calibration.scale
                pA_signal = (raw_signal + offset) * scale

                signal_segments = get_signal_segments_pA(pA_signal, move_table, stride, ts)
                if len(signal_segments) != len(query_seq):
                    continue

                signal_str_list = ["*".join(f"{x:.3f}" for x in seg) for seg in signal_segments]
                quality_str = "|".join(map(str, qual))

                results.append({
                    "read_id": read_id,
                    "ref_name": ref_name,
                    "mapped_start": ref_start_0b + 1,
                    "mapped_end": ref_end_0b,
                    "query_seq": query_seq,
                    "ref_seq_full": ref_seq_full,
                    "quality_str": quality_str,
                    "signal_str": "|".join(signal_str_list)
                })

        ref_fasta.close()
        for res in results:
            output_queue.put(res)

    except Exception as e:
        print(f"Error processing {pod5_path}: {e}", file=sys.stderr)


def worker(pod5_file, args, output_queue, bam_dict):
    process_pod5_file(pod5_file, bam_dict, args.reference, output_queue, clip=int(args.clip))


def main():
    parser = argparse.ArgumentParser(description='Extract signal from POD5 using coordinate-sorted BAM.')
    parser.add_argument('-o', '--output', required=True, help="Output TSV file.")
    parser.add_argument('-p', '--process', type=int, default=1, help='Number of processes.')
    parser.add_argument('--clip', type=int, default=10, help='Reserved (not used).')
    parser.add_argument('--pod5', required=True, help='Directory containing POD5 files.')
    parser.add_argument('-r', '--reference', required=True, help='Reference genome FASTA file.')
    parser.add_argument('--bam', required=True, help='Sorted and indexed BAM file.')
    args = parser.parse_args()

    if not os.path.exists(args.bam + ".bai"):
        raise FileNotFoundError(f"BAM index {args.bam}.bai not found. Run 'samtools index {args.bam}'")

    print("Loading BAM alignments into memory...")
    bam_dict = {}
    with pysam.AlignmentFile(args.bam, "rb") as bam:
        for aln in bam:
            if not aln.is_mapped or aln.is_secondary or aln.is_supplementary:
                continue
 
            bam_dict[aln.query_name] = {
                'ref_name': aln.reference_name,
                'ref_start': aln.reference_start,      # 0-based
                'ref_end': aln.reference_end,          # 0-based exclusive
                'query_seq': aln.query_sequence or "",
                'qual': list(aln.query_qualities) if aln.query_qualities else [0] * aln.query_length,
                'is_reverse': aln.is_reverse,
                'tags': dict(aln.get_tags())   
            }
    print(f"Loaded {len(bam_dict)} alignments.")

    pod5_files = glob.glob(os.path.join(args.pod5, '**/*.pod5'), recursive=True)
    print(f"Found {len(pod5_files)} POD5 files.")

    manager = multiprocessing.Manager()
    output_queue = manager.Queue()

    with multiprocessing.Pool(processes=args.process) as pool:
        worker_func = partial(
            worker,
            args=args,
            output_queue=output_queue,
            bam_dict=bam_dict 
        )
        list(tqdm(
            pool.imap_unordered(worker_func, pod5_files),
            total=len(pod5_files),
            desc="Processing POD5"
        ))


    results = []
    while not output_queue.empty():
        results.append(output_queue.get())

    print(f"Writing {len(results)} records to {args.output}...")
    with open(args.output, 'w') as out:
        out.write("read_id\tref_name\tinter_start\tquery_seq\tquality\tref_seq\tsignal\n")
        for res in results:
            line = f"{res['read_id']}\t{res['ref_name']}\t{res['mapped_start']}\t{res['query_seq']}\t{res['quality_str']}\t{res['ref_seq_full']}\t{res['signal_str']}\n"
            out.write(line)

    print("Done.")


if __name__ == "__main__":
    main()
