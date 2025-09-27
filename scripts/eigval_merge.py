import h5py
import os
import sys
import re
import numpy as np
from collections import defaultdict
import argparse
from tqdm import tqdm
import subprocess

def find_best_eigen(root_dir, target_B , target_Jpm , N, begin ):

    # total number of lowest eigenvalues to collect
    
    def parse_B_from_filename(filename):
        pattern = r"Jpm=([0-9.]+)%Bx=([0-9.]+)%By=([0-9.]+)%Bz=([0-9.]+)%"
        match = re.search(pattern, filename)
        if match:
            Jpm, Bx, By, Bz = map(float, match.groups())
            return Jpm, Bx, By, Bz
        return None
    
    # List to hold (eigenvalue, sector, index in file)
    eigen_list = []
    
    total = sum(1 for _ in os.walk(root_dir))

    for dirpath, dirnames, filenames in tqdm(os.walk(root_dir), total=total, desc="Walking dirs"):
    # for dirpath, dirnames, filenames in os.walk(root_dir):
        if not os.path.basename(dirpath).startswith(begin):
            continue
        if not re.search(begin+r"basis_s[\d.]+", dirpath):
            continue
        sector_match = re.search(begin+r"basis_(s[\d.]+)", dirpath)
        if not sector_match:
            continue
        sector = sector_match.group(1)
        print(sector,end='                  \r')
    
        for fname in filenames:
            if not fname.endswith(".eigs.h5"):
                continue
            parsed = parse_B_from_filename(fname)
            if not parsed:
                continue
            Jpm, Bx, By, Bz = parsed
            if not np.allclose((Bx, By, Bz), target_B):
                continue
            if np.abs(Jpm- target_Jpm) > 1e-10:
                continue
    
            filepath = os.path.join(dirpath, fname)
            with h5py.File(filepath, "r") as f:
                eigvals = f["/eigenvalues"][()]
                for idx, val in enumerate(eigvals):
                    eigen_list.append((val, sector, idx, filepath))
    
    # Sort all eigenvalues globally
    eigen_list.sort()

    selected_eigen = eigen_list[:N]
    return selected_eigen



def parse_args():
    parser = argparse.ArgumentParser(
            description="Find the lowest N eigenvalues across topological sectors."
            )
    parser.add_argument(
            "--root", type=str, default="../out/diag_DOQSI",
            help="Root directory containing sector folders"
            )
    parser.add_argument(
            "shape", type=str,
            help="Shape name (i.e. prefix of all out directories)"
            )
    parser.add_argument(
            "--Jpm", type=float, default=0.05,
            help="Transverse exchange"
            )
    parser.add_argument(
            "--B", type=float, default=(0., 0., 0.), nargs=3,
            help="Magnetic field B"
            )
    parser.add_argument(
            "-N", type=int, required=True,
            help="Total number of lowest eigenvalues to select"
            )
    parser.add_argument(
            "-d", "--dry_run", help="do not calculate expectations", action="store_true", default=False)
    return parser.parse_args()





if __name__ == "__main__":
    args=parse_args()


    selected_eigen = find_best_eigen(target_B =tuple(args.B),
                                     target_Jpm = args.Jpm, 
                                     N = args.N,
                                     root_dir = "../out/diag_DOQSI/",
                                     begin=args.shape
    )

    # Collect the lowest N
    sector_counts = defaultdict(int)
    file_lookup = {}
    
    for val, sector, idx, fname in selected_eigen:
        sector_counts[sector] += 1
        file_lookup[sector] = fname
    
    print()
    print("energy | sector")
    for e, sector, _, _ in selected_eigen:
        print(f"{e}\t{sector}")
    
    # Print result
    print("sector | number of eigenvalues")
    for sector, count in sector_counts.items():
        print(f"{sector} | {count}")
        if not args.dry_run:
            res = subprocess.run(
                ["build/eval_observables", file_lookup[sector], "-N", str(count)],
                capture_output=True, check=True
                )
            print(res.stdout)
            print(res.stderr)


