#!/usr/bin/env python3
"""
Benchmark script for bench_apply_mpi across HDF5 dataset sectors.
Cross-platform memory profiling (works on macOS and Linux).
Fits a linear model: memory = bytes_per_element * basis_size + overhead_kb
"""

import argparse
import csv
import subprocess
import sys
import time
import os
import resource
import tempfile
import threading
from pathlib import Path

import numpy as np

# ── optional h5py for dataset listing ────────────────────────────────────────
try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


# ── memory sampling ───────────────────────────────────────────────────────────

def _sample_peak_memory_kb(pid: int, stop_event: threading.Event, result: list):
    """Poll /proc/<pid>/status (Linux) or ps (macOS) every 50 ms, record peak RSS."""
    peak_kb = 0
    while not stop_event.is_set():
        try:
            if sys.platform == "linux":
                status = Path(f"/proc/{pid}/status").read_text()
                for line in status.splitlines():
                    if line.startswith("VmRSS:"):
                        kb = int(line.split()[1])
                        peak_kb = max(peak_kb, kb)
                        break
            else:  # macOS / BSD — use ps
                out = subprocess.check_output(
                    ["ps", "-o", "rss=", "-p", str(pid)],
                    stderr=subprocess.DEVNULL,
                )
                kb = int(out.strip())
                peak_kb = max(peak_kb, kb)
        except (FileNotFoundError, subprocess.CalledProcessError, ValueError):
            pass
        stop_event.wait(timeout=0.05)
    result.append(peak_kb)


def run_with_profiling(cmd: list[str]) -> dict:
    """
    Run *cmd*, capturing stdout+stderr and peak RSS (kB).
    Returns dict with keys: output, peak_memory_kb, time_real, time_user, time_sys.
    """
    t0 = time.perf_counter()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    stop_event = threading.Event()
    mem_result: list[int] = []
    monitor = threading.Thread(
        target=_sample_peak_memory_kb,
        args=(proc.pid, stop_event, mem_result),
        daemon=True,
    )
    monitor.start()

    output, _ = proc.communicate()

    stop_event.set()
    monitor.join(timeout=1)

    t1 = time.perf_counter()

    # Fallback: use rusage of this process's children (less accurate but always available)
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    time_user = usage.ru_utime
    time_sys  = usage.ru_stime

    peak_kb = mem_result[0] if mem_result else 0
    if peak_kb == 0:
        # Last-resort fallback via ru_maxrss
        ru_kb = usage.ru_maxrss
        if sys.platform == "darwin":
            ru_kb //= 1024   # macOS returns bytes
        peak_kb = ru_kb

    return {
        "output":         output,
        "peak_memory_kb": peak_kb,
        "time_real":      t1 - t0,
        "time_user":      time_user,
        "time_sys":       time_sys,
    }


# ── HDF5 dataset listing ──────────────────────────────────────────────────────

def list_datasets_h5py(hdf5_file: str) -> list[tuple[str, int]]:
    """Return [(dataset_path, n_rows), ...] using h5py."""
    results = []
    def _visitor(name, obj):
        if isinstance(obj, h5py.Dataset) and obj.ndim >= 1:
            results.append((f"/{name}", obj.shape[0]))
    with h5py.File(hdf5_file, "r") as f:
        f.visititems(_visitor)
    return results


def list_datasets_h5ls(hdf5_file: str) -> list[tuple[str, int]]:
    """Fallback: parse h5ls -r output for dataset paths and first dimension."""
    out = subprocess.check_output(["h5ls", "-r", hdf5_file], text=True)
    results = []
    for line in out.splitlines():
        if not line.startswith("/"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        path = parts[0]
        # shape field looks like  {5912/Inf, 2}
        shape_str = " ".join(parts[1:])
        try:
            first_dim = int(shape_str.lstrip("{").split("/")[0].split(",")[0])
            results.append((path, first_dim))
        except ValueError:
            pass
    return results


def get_datasets(hdf5_file: str) -> list[tuple[str, int]]:
    if HAS_H5PY:
        return list_datasets_h5py(hdf5_file)
    return list_datasets_h5ls(hdf5_file)


# ── linear fit ────────────────────────────────────────────────────────────────

def linear_fit(sizes: list[int], memories_kb: list[int]):
    """Fit memory_kb = m * size + b via numpy least squares."""
    x = np.array(sizes, dtype=float)
    y = np.array(memories_kb, dtype=float)
    A = np.column_stack([x, np.ones_like(x)])
    result = np.linalg.lstsq(A, y, rcond=None)
    m, b = result[0]
    bytes_per_element = m * 1024   # kB → bytes
    overhead_kb       = b
    return bytes_per_element, overhead_kb


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark bench_apply_mpi across HDF5 sectors."
    )
    parser.add_argument("N",         type=int, nargs="?", default=50,
                        help="Number of datasets to process (default: 50)")
    parser.add_argument("NRANKS",    type=int, nargs="?", default=1,
                        help="MPI ranks (default: 1)")
    parser.add_argument("--hdf5",    default="../lattice_files/pyro_2,0,0_0,3,0_0,0,4.0.basis.partitioned.h5")
    parser.add_argument("--json",    default="../lattice_files/pyro_2,0,0_0,3,0_0,0,4.json")
    parser.add_argument("--max-size", type=int, default=80_000)
    parser.add_argument("--min-size", type=int, default=10_000)
    parser.add_argument("--output",  default=None,
                        help="Output CSV path (default: results_n<NRANKS>.csv)")
    args = parser.parse_args()

    N        = args.N
    NRANKS   = args.NRANKS
    HDF5     = args.hdf5
    JSON     = args.json
    OUT_CSV  = args.output or f"results_n{NRANKS}.csv"

    print(f"Processing up to {N} datasets with {NRANKS} rank(s)")
    print(f"  HDF5 : {HDF5}")
    print(f"  JSON : {JSON}")
    print(f"  CSV  : {OUT_CSV}")

    datasets = get_datasets(HDF5)
    print(f"Found {len(datasets)} datasets in HDF5 file")

    fit_sizes:   list[int]   = []
    fit_mem_kb:  list[int]   = []
    success = 0

    with open(OUT_CSV, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "basis_size", "max_memory_kb",
            "time_real_s", "time_user_s", "time_sys_s",
            "dataset_name",
        ])

        for dataset, size in datasets:
            if success >= N:
                break

            if not (args.min_size <= size <= args.max_size):
                continue

            print(f"  [{success+1}/{N}] {dataset}  (size={size})")

            cmd = [
                "mpirun", "-n", str(NRANKS),
                "build/bench/bench_apply_mpi",
                JSON,
                "--sector", dataset,
                "--trim",
            ]

            try:
                prof = run_with_profiling(cmd)
            except FileNotFoundError as e:
                print(f"    ERROR: {e} — skipping")
                continue

            writer.writerow([
                size,
                prof["peak_memory_kb"],
                f"{prof['time_real']:.4f}",
                f"{prof['time_user']:.4f}",
                f"{prof['time_sys']:.4f}",
                dataset,
            ])
            csvfile.flush()

            fit_sizes.append(size)
            fit_mem_kb.append(prof["peak_memory_kb"])
            success += 1

    print(f"\nDone. Processed {success} datasets → {OUT_CSV}")

    # ── linear fit ──────────────────────────────────────────────────────────
    if len(fit_sizes) >= 2:
        bpe, overhead_kb = linear_fit(fit_sizes, fit_mem_kb)
        print("\n── Linear fit: memory = m·basis_size + b ──────────────────")
        print(f"  Bytes per basis element : {bpe:,.1f} B/element")
        print(f"  Overhead                : {overhead_kb:,.1f} kB")
        if bpe > 0:
            print(f"  (≈ {bpe/8:,.2f} float64 values per basis element)")
    else:
        print("\nNot enough data points for a linear fit (need ≥ 2).")


if __name__ == "__main__":
    main()
