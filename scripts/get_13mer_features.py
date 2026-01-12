import os
import re
import random
import argparse
import numpy as np

from statsmodels import robust
import faulthandler
faulthandler.enable()


def reverse_compliment(seq):
    """
    Generate reverse complement sequence.

    Args:
        seq (str): Origin sequence.

    Returns:
        str: Reverse complement sequence.
    """
    seq=seq.replace("A","t")
    seq=seq.replace("G","c")
    seq=seq.replace("C","g")
    seq=seq.replace("T","a")
    seq=seq.replace("N","N")
    return seq[::-1].upper()

def interp(signal):
    if not isinstance(signal, (list, np.ndarray)):
       pass
    arr = np.asarray(signal, dtype=np.float32)

    n = arr.size
    if n == 0:
        return [0.0] * 100
    if n == 1:
        v = float(arr[0])
        return [round(v, 4)] * 100


    if not np.all(np.isfinite(arr)):
        for i in range(1, n):
            if not np.isfinite(arr[i]):
                arr[i] = arr[i - 1]
        for i in range(n - 2, -1, -1):
            if not np.isfinite(arr[i]):
                arr[i] = arr[i + 1]

    x_old = np.linspace(0.0, n - 1.0, n, dtype=np.float32)
    x_new = np.linspace(0.0, n - 1.0, 100, dtype=np.float32)
    y = np.interp(x_new, x_old, arr, left=arr[0], right=arr[-1]).astype(np.float32, copy=False)
    return np.round(y, 4).tolist()

def signal_to_file2(out_dir,file_name,line):
    """
    Append the line to a file in the specified directory.
    
    Args:
        out_dir (str): Output directory path.
        file_name (str): Name of the file to append the line to.
        line (str): Line of text to be written to the file.
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    with open(out_dir+"/"+args.label,"a") as f:
            f.writelines(line)

def signal_to_file(out_dir,file_name,line):
    """
    Write the line to a file in the specified directory based on a random selection.
    
    Args:
        out_dir (str): Output directory path.
        file_name (str): Name of the file to be created.
        line (str): Line of text to be written to the file.
    """
    if not os.path.exists(out_dir+"/test/"+args.label):
        os.makedirs(out_dir+"/test/"+args.label)
    if not os.path.exists(out_dir+"/train/"+args.label):
        os.makedirs(out_dir+"/train/"+args.label)
    if random.random() <= 0.2:
        with open(out_dir+"/test/"+args.label+"/"+file_name,"w") as f:
            f.writelines(line)
    else:
        with open(out_dir+"/train/"+args.label+"/"+file_name,"w") as f: 
            f.writelines(line)


def convert_base_name(base_name):
    """
    Converts a base name into a regular expression pattern.

    Args:
        base_name (str): Input base name to be converted.

    Returns:
        str: Regular expression pattern representing the converted base name.
    """
    merge_bases = {
        'A': 'A',
        'C': 'C',
        'G': 'G',
        'T': 'T',
        'M': '[AC]',
        'V': '[ACG]',
        'R': '[AG]',
        'H': '[ACT]',
        'W': '[AT]',
        'D': '[AGT]',
        'S': '[CG]',
        'B': '[CGT]',
        'Y': '[CT]',
        'N': '[ACGT]',
        'K': '[GT]'
    }
    pattern = ''
    for base in base_name:
        pattern += merge_bases.get(base, base)
    return pattern

def safe_mad(x: np.ndarray) -> float:
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    if not np.isfinite(mad) or mad == 0.0:
        mad = float(x.std())
        if not np.isfinite(mad) or mad == 0.0:
            mad = 1.0
    else:
        mad *= 1.4826 
    return mad 
def _safe_stats(arr: np.ndarray):
 
    n = arr.size
    if n == 0:
        return 0.0, 0.0, 0.0, 0
   
    if n > 200_000:
        arr = arr[::4] 
        n = arr.size
        if n == 0:
            return 0.0, 0.0, 0.0, 0
    
    m  = float(np.mean(arr, dtype=np.float64))
    sd = float(np.std(arr,  dtype=np.float64))
    med= float(np.median(arr))
    return m, sd, med, int(n)

def extract_13mer_features(signal_file):
    """
    Extracts 5-mer centered features from a signal file.
    
    Args:
        signal_file (str): Path to the signal file.
    """
 
    max_features_per_file = 400000  
    file_index = 1                  
    feature_count = 0          


    base_output = os.path.basename(args.output) 
    base_name, ext = os.path.splitext(base_output)

    current_output = f"{base_name}_{file_index}{ext}"
    out = open(os.path.join(os.path.dirname(args.output), current_output), "w")

    base_quality_dict = dict()


    kmer_filter = convert_base_name("......C......")  

    clip = int(args.clip) if hasattr(args, "clip") else 6
    count = 0
    scaling = "median_mad"
    
    with open(signal_file) as f:
        next(f)
        for line_num, line in enumerate(f, 1):
            try:
                line = line.rstrip()
                items = line.split("\t")
                required_columns = 7 
                if len(items) < required_columns:
                    pass
                    continue
                read_id = items[0]
                chr = items[1]
                start = int(items[2])
                reference_sequence = items[3]
                base_quality_list = items[4].split("|")
                sequence = line.split("\t")[5]
                
                if len(sequence) < 500:
                    continue
                
                signal_string = line.split("\t")[6]
                raw_signal = [
                    np.fromstring(seg, sep='*', dtype=np.float32)
                    for seg in signal_string.split('|')
                ]
                
       
                full_length_signal = np.fromstring(
                signal_string.replace('|', ' ').replace('*', ' '),
                sep=' ',
                dtype=np.float32
                    )       
                fl_min = float(full_length_signal.min())
                fl_max = float(full_length_signal.max())
                fl_mean = float(full_length_signal.mean())
                fl_std  = float(full_length_signal.std())

                for index in range(clip, len(sequence)-clip):
                    center_pos = index
                    kmer_sequence = sequence[center_pos-6:center_pos+7]
                    if len(kmer_sequence) != 13:
                        continue
       
                    if kmer_sequence[6] != 'C': 
                        continue
          
                    if not re.search(kmer_filter, kmer_sequence):
                        continue
                  
                    ref_kmer = reference_sequence[center_pos-6:center_pos+7] 
                    if len(ref_kmer) != 13:
                        continue 
                    if kmer_sequence != ref_kmer:
                        continue

                    kmer_raw_signal = raw_signal[center_pos-6:center_pos+7]  
                   
                    if any(x.size == 0 for x in kmer_raw_signal):
                        continue

                    if scaling == "min_max":
                        denom = (fl_max - fl_min) if fl_max > fl_min else 1.0
                        kmer_raw_signal = [(x - fl_min) / denom for x in kmer_raw_signal]
                    elif scaling == "zscore":
                        denom = fl_std if fl_std > 0 else 1.0
                        kmer_raw_signal = [(x - fl_mean) / denom for x in kmer_raw_signal]
                    elif scaling == "median_mad":
                        fl_med = float(np.median(full_length_signal))
                        fl_mad = safe_mad(full_length_signal)
                        kmer_raw_signal = [(x - fl_med) / fl_mad for x in kmer_raw_signal]
              
                    _stats = [_safe_stats(x) for x in kmer_raw_signal]
                    mean   = [np.round(s[0], 3) for s in _stats]
                    std    = [np.round(s[1], 3) for s in _stats]
                    median = [np.round(s[2], 3) for s in _stats]
                    length = [s[3]           for s in _stats]
                    kmer_base_quality = base_quality_list[center_pos-6:center_pos+7]  

                    for i in range(13): 
                        kmer_raw_signal[i] = interp(kmer_raw_signal[i])

                    line_fields = [
                        read_id,
                        chr,
                        str(start + center_pos),
                        kmer_sequence,
                        "|".join(map(str, mean)),
                        "|".join(map(str, std)),
                        "|".join(map(str, median)),
                        "|".join(map(str, length)),
                        "|".join(map(str, kmer_base_quality))
                    ]
             
                    for pos_signal in kmer_raw_signal:
                        line_fields.append("|".join(map(str, pos_signal)))
                 
                    out.write(line_fields[0])
                    for field in line_fields[1:]:
                         out.write('\t')
                         out.write(field)
                    out.write('\n')

                    file_name = f"{read_id}_{chr}_{start+center_pos}_{kmer_sequence}.feature"
                    count += 1
                    feature_count += 1
              
                    if count % 10000 == 0:
                        print(f"Processed {count} features")
                    if feature_count >= max_features_per_file:
                        out.close()
                        file_index += 1
                        feature_count = 0
                        current_output = f"{base_name}_{file_index}{ext}"
                        out = open(os.path.join(os.path.dirname(args.output), current_output), "w")
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                   
                
    out.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract feature from signal.')

    parser.add_argument('--signal_file', required = True, help='\tSignal file')
    parser.add_argument('--clip', default=6, help='\tBase clip at both ends')  
    parser.add_argument('-o','--output', required = True, help="\tOutput file.")
    parser.add_argument('--motif', required = True, help="\tSequence motif")

    args = parser.parse_args()
    
    extract_13mer_features(args.signal_file)  
