import os, csv, re, math, hashlib
import pandas as pd


DATA_FILES = {
    'CpG': {
        'neg': 'all_neg_CpG.csv',
        'pos': 'all_pos_CpG.csv'
    }
}
NUM_SIGNALS   = 13
TARGET_TOTAL  = 400000
RATIOS        = (0.8, 0.1, 0.1)
TSV_SEP_IN    = '\t'
CSV_SEP_OUT   = ','
VAR_AS_STD    = False
CHUNK_SIZE    = 100_000
TMP_DIR       = "_tmp_stream_norm"

OUTPUT_COLUMNS = (
    ['k_mer', 'signal_means', 'signal_stds', 'signal_lens'] +
    [f'signal{i}' for i in range(1, NUM_SIGNALS + 1)] +
    ['methy_lable']
)


def _pipe_to_comma(s: str) -> str:
    if pd.isna(s): return ''
    s = str(s)
    s = s.replace('，', ',').replace('"', '').replace("'", '')
    parts = re.split(r'[|,]', s)
    parts = [p.strip() for p in parts if p.strip() != '']
    return ','.join(parts)

def _pipe_to_comma_and_sqrt(s: str) -> str:
    if pd.isna(s): return ''
    s = str(s).replace('"', '').replace("'", '')
    out = []
    for p in s.split('|'):
        p = p.strip()
        if not p: continue
        try:
            out.append(str(math.sqrt(float(p))))
        except Exception:
            out.append(p)
    return ','.join(out)

def _read_header_cols(path: str):
    df0 = pd.read_csv(path, sep=TSV_SEP_IN, dtype=str, engine='python', nrows=0)
    return list(df0.columns)

def _normalize_chunk(df: pd.DataFrame, label_value: int) -> pd.DataFrame:
    o = pd.DataFrame()
    o['k_mer']        = df['sequence'].astype(str)
    o['signal_means'] = df['current_mean'].map(_pipe_to_comma)
    if 'current_std' in df.columns:
        o['signal_stds'] = df['current_std'].map(_pipe_to_comma)
    else:
        o['signal_stds'] = df['current_var'].map(
            _pipe_to_comma_and_sqrt if VAR_AS_STD else _pipe_to_comma
        )
    o['signal_lens']  = df['current_length'].map(_pipe_to_comma)
    for i in range(1, NUM_SIGNALS + 1):
        o[f'signal{i}'] = df[f'current_signal{i}'].map(_pipe_to_comma)
    o['methy_lable']  = int(label_value)
    return o[OUTPUT_COLUMNS]

def _normalize_file_to_tmp(path: str, label_value: int, out_csv: str):
    os.makedirs(TMP_DIR, exist_ok=True)
    header = _read_header_cols(path)

    needed = ['sequence','current_mean','current_length'] + \
             [f'current_signal{i}' for i in range(1, NUM_SIGNALS+1)]
    if 'current_std' in header:
        needed.append('current_std')
    elif 'current_var' in header:
        needed.append('current_var')
    else:
        raise KeyError("Missing required column: either 'current_std' or 'current_var' must be present")

    missing = [c for c in needed if c not in header]
    if missing:
        raise KeyError(f"{path} is missing required columns: {missing}")

    first = True
    with open(out_csv, 'w', newline='') as fo:
        writer = csv.writer(fo)
        writer.writerow(OUTPUT_COLUMNS)
    for chunk in pd.read_csv(
        path, sep=TSV_SEP_IN, dtype=str, engine='python',
        usecols=needed, chunksize=CHUNK_SIZE
    ):
        out = _normalize_chunk(chunk, label_value)
        out.to_csv(out_csv, mode='a', index=False, header=False)

def _line_count(path: str) -> int:
    n = 0
    with open(path, 'r') as f:
        _ = f.readline()  # skip header
        for _ in f: n += 1
    return n

def _stable_split_choice(key_str: str):
    h = hashlib.blake2b(key_str.encode('utf-8'), digest_size=8).digest()
    val = int.from_bytes(h, 'little') / float(1<<64)
    if val < RATIOS[0]:
        return 0  # train
    elif val < RATIOS[0] + RATIOS[1]:
        return 1  # val
    else:
        return 2  # test

def _open_writers():
    f_train = open('train.csv', 'w', newline='')
    f_val   = open('val.csv',   'w', newline='')
    f_test  = open('test.csv',  'w', newline='')
    wt, wv, ws = csv.writer(f_train), csv.writer(f_val), csv.writer(f_test)
    for w in (wt,wv,ws): w.writerow(OUTPUT_COLUMNS)
    return (f_train, f_val, f_test), (wt, wv, ws)

def _close_files(files):
    for f in files:
        try: f.close()
        except Exception: pass

def main():
    os.makedirs(TMP_DIR, exist_ok=True)

    tmp_pos = os.path.join(TMP_DIR, 'pos_norm.csv')
    tmp_neg = os.path.join(TMP_DIR, 'neg_norm.csv')

    for ctx, pair in DATA_FILES.items():
        _normalize_file_to_tmp(pair['pos'], 1, tmp_pos)
        _normalize_file_to_tmp(pair['neg'], 0, tmp_neg)

    pos_total = _line_count(tmp_pos)
    neg_total = _line_count(tmp_neg)
    cap_each  = min(pos_total, neg_total)

    tgt_train = int(TARGET_TOTAL * RATIOS[0])
    tgt_val   = int(TARGET_TOTAL * RATIOS[1])
    tgt_test  = int(TARGET_TOTAL * RATIOS[2])

    max_total = 2 * cap_each
    wanted    = tgt_train + tgt_val + tgt_test
    if wanted > max_total:
        scale = max_total / wanted
        tgt_train = int(round(tgt_train * scale))
        tgt_val   = int(round(tgt_val   * scale))
        tgt_test  = int(round(tgt_test  * scale))

    half = lambda n: (n//2, n - n//2)  # (neg, pos)
    q_train_neg, q_train_pos = half(tgt_train)
    q_val_neg,   q_val_pos   = half(tgt_val)
    q_test_neg,  q_test_pos  = half(tgt_test)

    quotas = {
        0: {0: q_train_neg, 1: q_train_pos},  # train
        1: {0: q_val_neg,   1: q_val_pos},    # val
        2: {0: q_test_neg,  1: q_test_pos},   # test
    }
    remain_total = {0: tgt_train, 1: tgt_val, 2: tgt_test}

    files, writers = _open_writers()
    wt, wv, ws = writers
    ws_by_idx = {0: wt, 1: wv, 2: ws}

    def assign_and_write(row: list, label: int, keystr: str):
        start = _stable_split_choice(keystr)
        for off in range(3):
            idx = (start + off) % 3  # 0,1,2 correspond to train/val/test
            if quotas[idx][label] > 0 and remain_total[idx] > 0:
                ws_by_idx[idx].writerow(row)
                quotas[idx][label] -= 1
                remain_total[idx]  -= 1
                return True
        return False

    def _row_iter(path: str):
        with open(path, 'r') as f:
            r = csv.reader(f)
            _ = next(r, None)  # skip header
            for row in r:
                yield row

    it_pos = _row_iter(tmp_pos)
    it_neg = _row_iter(tmp_neg)

    wrote_pos = wrote_neg = 0
    turn = 0  # 0 -> pos, 1 -> neg

    def need_more():
        return any(v > 0 for v in remain_total.values())

    while need_more() and (wrote_pos < cap_each or wrote_neg < cap_each):
        if turn == 0 and wrote_pos < cap_each:
            try:
                row = next(it_pos)
            except StopIteration:
                turn = 1
                continue

            key = (row[0] or '') + '|' + (row[4] if len(row) > 4 else '')
            ok = assign_and_write(row, label=1, keystr=key)
            if ok: wrote_pos += 1
            turn = 1
        else:
            if wrote_neg >= cap_each:
                turn = 0
                continue
            try:
                row = next(it_neg)
            except StopIteration:
                turn = 0
                continue
            key = (row[0] or '') + '|' + (row[4] if len(row) > 4 else '')
            ok = assign_and_write(row, label=0, keystr=key)
            if ok: wrote_neg += 1
            turn = 0

    _close_files(files)

    print(f"Completed: train/val/test = {sum(quotas[0].values()) == 0}/{sum(quotas[1].values()) == 0}/{sum(quotas[2].values()) == 0}")
    print("Output files: train.csv, val.csv, test.csv")
    print(f"(Intermediate files saved to {TMP_DIR}/ , you may inspect and delete manually)")


if __name__ == "__main__":
    main()
