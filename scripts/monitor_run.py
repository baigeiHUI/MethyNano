import argparse
import csv
import os
import subprocess
import time
from typing import List

import psutil


def get_descendant_processes(proc: psutil.Process) -> List[psutil.Process]:
    try:
        children = proc.children(recursive=True)
        return [proc] + children
    except psutil.Error:
        return [proc]


def get_total_rss_mb(proc: psutil.Process) -> float:
    total = 0
    for p in get_descendant_processes(proc):
        try:
            total += p.memory_info().rss
        except psutil.Error:
            pass
    return total / (1024 ** 2)


def get_total_cpu_percent(proc: psutil.Process) -> float:
    total = 0.0
    for p in get_descendant_processes(proc):
        try:
            total += p.cpu_percent(interval=None)
        except psutil.Error:
            pass
    return total


def get_gpu_mem_mb_by_pids(pids: List[int]) -> float:
    if not pids:
        return 0.0
    pid_set = {str(pid) for pid in pids}
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return 0.0

        total = 0.0
        for line in result.stdout.strip().splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) != 2:
                continue
            pid, mem = parts
            if pid in pid_set:
                try:
                    total += float(mem)
                except ValueError:
                    pass
        return total
    except FileNotFoundError:
        return 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=float, default=1.0, help="Sampling interval in seconds.")
    parser.add_argument("--log", required=True, help="Output CSV file.")
    parser.add_argument("cmd", nargs=argparse.REMAINDER, help="Command to run.")
    args = parser.parse_args()

    if not args.cmd:
        raise ValueError("Provide a command after --log, for example: python train.py")

    cmd = args.cmd
    if cmd[0] == "--":
        cmd = cmd[1:]

    os.makedirs(os.path.dirname(os.path.abspath(args.log)) or ".", exist_ok=True)

    start_time = time.time()
    proc_sub = subprocess.Popen(cmd)
    proc = psutil.Process(proc_sub.pid)

    for p in get_descendant_processes(proc):
        try:
            p.cpu_percent(interval=None)
        except psutil.Error:
            pass

    peak_rss = 0.0
    peak_gpu = 0.0

    with open(args.log, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp_sec",
            "parent_pid",
            "num_procs",
            "rss_mb",
            "cpu_percent",
            "gpu_mem_mb",
        ])

        while True:
            alive = proc_sub.poll() is None

            try:
                procs = get_descendant_processes(proc)
                alive_procs = []
                for p in procs:
                    try:
                        if p.is_running():
                            alive_procs.append(p)
                    except psutil.Error:
                        pass

                pids = [p.pid for p in alive_procs]
                rss_mb = get_total_rss_mb(proc) if alive_procs else 0.0
                cpu_percent = get_total_cpu_percent(proc) if alive_procs else 0.0
                gpu_mem_mb = get_gpu_mem_mb_by_pids(pids)

                elapsed = time.time() - start_time
                writer.writerow([
                    round(elapsed, 3),
                    proc_sub.pid,
                    len(alive_procs),
                    round(rss_mb, 3),
                    round(cpu_percent, 3),
                    round(gpu_mem_mb, 3),
                ])
                f.flush()

                peak_rss = max(peak_rss, rss_mb)
                peak_gpu = max(peak_gpu, gpu_mem_mb)

            except psutil.Error:
                pass

            if not alive:
                break

            time.sleep(args.interval)

    total_time = time.time() - start_time
    print(f"Exit code: {proc_sub.returncode}")
    print(f"Total time: {total_time:.2f} sec")
    print(f"Peak RSS memory: {peak_rss:.2f} MB")
    print(f"Peak GPU memory: {peak_gpu:.2f} MB")
    print(f"Log saved to: {args.log}")


if __name__ == "__main__":
    main()
