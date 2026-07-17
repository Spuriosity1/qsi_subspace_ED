#!/usr/bin/env python3
"""Drive diag_DOQSI_fused over the irreducible field-direction triangle.

This produces a 3D magnetic-field phase diagram at fixed J_pm by diagonalising
the DOQSI Hamiltonian for a grid of field *directions*, on one
constant-|B| shell.

By cubic (Oh) symmetry, every inequivalent field direction is represented once
inside the spherical triangle whose corners are the high-symmetry axes

    [100]  --  [110]  --  [111]

so we only sweep that wedge and let symmetry fill in the rest of the sphere.
For each sampled direction we run the field on each requested shell |B| (here
0.2 and 0.3, in units of Jzz), giving field vectors  B = |B| * n_hat.

The heavy lifting -- basis enumeration, Lanczos, and ring-observable evaluation
-- is done by the compiled `diag_DOQSI_fused_mpi` binary; this script only
generates the (J_pm, B) grid, launches the binary once per point (skipping
points whose output already exists), then reads the resulting HDF5 files and
renders the phase diagram.

Examples
--------
    # full run + plot, 4 MPI ranks
    ./phasedia_3d_fused.py --np 4

    # just re-draw from existing output, colour by ring order parameter
    ./phasedia_3d_fused.py --plot-only --observable ring_abs

    # see the command list without running anything
    ./phasedia_3d_fused.py --dry-run
"""

import argparse
import os
import subprocess
import sys

import numpy as np

# --------------------------------------------------------------------------
# Locations (resolved relative to this script so it works from any cwd)
# --------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)                       # qsi_subspace_ED/
PROJECT_ROOT = os.path.dirname(REPO_ROOT)                     # 010_qsi_subspace_ED/

DEFAULT_BIN = os.path.join(PROJECT_ROOT, "bin", "diag_DOQSI_fused_mpi")
DEFAULT_LATTICE = os.path.join(
    REPO_ROOT, "test", "lattice_files", "pyro_2,0,0_0,2,0_0,0,2.json")
DEFAULT_OUTDIR = os.path.join(PROJECT_ROOT, "out", "phasedia_3d")

# Corners of the irreducible field-direction triangle (unnormalised).
CORNERS = {
    "100": np.array([1.0, 0.0, 0.0]),
    "110": np.array([1.0, 1.0, 0.0]),
    "111": np.array([1.0, 1.0, 1.0]),
}


# --------------------------------------------------------------------------
# Direction sampling
# --------------------------------------------------------------------------
def sample_triangle_directions(n_div):
    """Unit vectors filling the [100]-[110]-[111] spherical triangle.

    Barycentric subdivision of the flat triangle spanned by the three corner
    axes, each interior/edge point projected onto the unit sphere. `n_div` sets
    the resolution: n_div+1 points along each edge, (n_div+1)(n_div+2)/2 total.
    Duplicate directions (shared corners/edges collapse) are removed.
    """
    a = CORNERS["100"] / np.linalg.norm(CORNERS["100"])
    b = CORNERS["110"] / np.linalg.norm(CORNERS["110"])
    c = CORNERS["111"] / np.linalg.norm(CORNERS["111"])

    seen = {}
    dirs = []
    for i in range(n_div + 1):
        for j in range(n_div + 1 - i):
            k = n_div - i - j
            v = (i * a + j * b + k * c) / n_div
            v = v / np.linalg.norm(v)
            key = tuple(np.round(v, 9))
            if key in seen:
                continue
            seen[key] = True
            dirs.append(v)
    return np.array(dirs)


def out_basename(jpm, B):
    """Reproduce diag_DOQSI_fused_mpi's observable filename for a (Jpm, B) point.

    The binary formats the stem as "Jpm=%.4f%%Bx=%.4f%%By=%.4f%%Bz=%.4f%%"
    (each %% is a literal '%') and writes ring observables to <stem>.eigs.out.h5.
    """
    stem = (f"Jpm={jpm:.4f}%Bx={B[0]:.4f}%By={B[1]:.4f}%Bz={B[2]:.4f}%")
    return stem + ".eigs.out.h5"


# --------------------------------------------------------------------------
# Running the binary
# --------------------------------------------------------------------------
def build_command(args, B):
    cmd = []
    if args.np > 1 or args.force_mpirun:
        cmd += [args.mpirun, "-np", str(args.np)]
    cmd += [
        args.binary,
        args.lattice,
        str(args.n_spinon_pairs),
        "--Jpm", str(args.jpm),
        "--B", f"{B[0]:.6g}", f"{B[1]:.6g}", f"{B[2]:.6g}",
        "-o", args.outdir,
        "-k", str(args.ncv),
        "--rtol", str(args.rtol),
        "--atol", str(args.atol),
    ]
    return cmd


def run_grid(args, field_points):
    os.makedirs(args.outdir, exist_ok=True)
    n = len(field_points)
    for idx, (B, mag, _n_hat) in enumerate(field_points, 1):
        outfile = os.path.join(args.outdir, out_basename(args.jpm, B))
        tag = f"[{idx}/{n}] |B|={mag:.2f}  B=({B[0]:.4f},{B[1]:.4f},{B[2]:.4f})"

        if os.path.exists(outfile) and not args.force:
            print(f"{tag}  -> exists, skipping")
            continue

        cmd = build_command(args, B)
        if args.dry_run:
            print(" ".join(cmd))
            continue

        print(f"{tag}  -> running")
        ret = subprocess.run(cmd)
        if ret.returncode != 0:
            print(f"  WARNING: binary exited {ret.returncode} for {tag}",
                  file=sys.stderr)


# --------------------------------------------------------------------------
# Reading observables
# --------------------------------------------------------------------------
def scalar_from_h5(path, observable):
    """Reduce one .eigs.out.h5 file to a single colour value."""
    import h5py

    with h5py.File(path, "r") as f:
        if observable == "energy":
            return float(np.asarray(f["/eigenvalues"][()]).ravel()[0])
        if observable == "flippability":
            return float(np.mean(np.asarray(f["/flippability"][()]).ravel()))
        if observable == "ring":
            return float(np.mean(np.asarray(f["/ring"][()]).ravel()))
        if observable == "ring_abs":
            return float(np.mean(np.abs(np.asarray(f["/ring"][()]).ravel())))
    raise ValueError(f"unknown observable {observable!r}")


def collect_results(args, field_points):
    rows = []
    missing = 0
    for B, mag, n_hat in field_points:
        outfile = os.path.join(args.outdir, out_basename(args.jpm, B))
        if not os.path.exists(outfile):
            missing += 1
            continue
        try:
            val = scalar_from_h5(outfile, args.observable)
        except (OSError, KeyError) as exc:
            print(f"  WARNING: could not read {outfile}: {exc}", file=sys.stderr)
            missing += 1
            continue
        rows.append((B, mag, n_hat, val))
    if missing:
        print(f"  note: {missing} field point(s) had no readable output")
    return rows



def from_vector3(B_3d):
    # Projects 3D vector to 111 projection of sphere
    B_3d = np.array(B_3d).T
    B_3d = (B_3d / np.linalg.norm(B_3d, axis=0)).T

    # plane normal
    n = np.array([1,1,1])
    x0 = np.array([2,2,2]) # observer position

    lam = - x0 @ n / (B_3d - x0) @ n

    x = x0 + (lam*(B_3d - x0).T).T
    
    R6=1/np.sqrt(6)
    R2=1/np.sqrt(2)
    return np.array([[R6,R6,-2*R6],[R2,-R2,0]]) @ x.T # project onto the plane


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------
def plot_phase_diagram(args, rows):
    if not rows:
        print("Nothing to plot (no results found).", file=sys.stderr)
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    B = np.array([r[0] for r in rows])
    vals = np.array([r[3] for r in rows])

    fig, ax = plt.subplots(figsize=(3, 3))

    X, Y = from_vector3(B)

    sc = ax.scatter(X,Y, c=vals, cmap="viridis"
                    )
    cb = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
    cb.set_label(args.observable)

    # annotate the three high-symmetry axes on the outer shell
    # rmax = max(m for _, m, _, _ in rows)
    # for name, axis in CORNERS.items():
    #     d = axis / np.linalg.norm(axis) * rmax
    #     ax.text(d[0], d[1], d[2], f"[{name}]", fontsize=10, color="crimson")
    #
    # ax.set_xlabel(r"$B_x/J_{zz}$")
    # ax.set_ylabel(r"$B_y/J_{zz}$")
    # ax.set_zlabel(r"$B_z/J_{zz}$")
    # ax.set_title(f"DOQSI field phase diagram  "
    #              f"($J_{{pm}}={args.jpm}$, |B| shells "
    #              f"{', '.join(f'{m:g}' for m in args.shells)})")
    # try:
    #     ax.set_box_aspect((1, 1, 1))
    # except Exception:
    #     pass

    fig.tight_layout()
    fig.savefig(args.figure, dpi=150)
    print(f"Wrote figure to {args.figure}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def get_parser():
    p = argparse.ArgumentParser(
        description="3D field phase diagram driver for diag_DOQSI_fused.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--lattice", default=DEFAULT_LATTICE,
                   help="lattice json spec")
    p.add_argument("--binary", default=DEFAULT_BIN,
                   help="path to diag_DOQSI_fused_mpi")
    p.add_argument("--outdir", "-o", default=DEFAULT_OUTDIR,
                   help="output directory for HDF5 results")

    p.add_argument("--jpm", type=float, default=-0.05, help="J_pm / Jzz")
    p.add_argument("--mag", type=float,  default=0.2,
                   help="constant field magnitude |B| / Jzz")
    p.add_argument("--n-div", type=int, default=6,
                   help="triangle subdivision (higher = finer direction grid)")
    p.add_argument("--n-spinon-pairs", type=int, default=0)

    # MPI / launcher
    p.add_argument("--np", type=int, default=1, help="MPI ranks per run")
    p.add_argument("--mpirun", default="mpirun", help="MPI launcher")
    p.add_argument("--force-mpirun", action="store_true",
                   help="use the launcher even when --np 1")

    # Lanczos knobs forwarded to the binary
    p.add_argument("--ncv", "-k", type=int, default=15)
    p.add_argument("--rtol", type=int, default=-8)
    p.add_argument("--atol", type=int, default=-8)

    # analysis
    p.add_argument("--observable", default="flippability",
                   choices=["flippability", "ring", "ring_abs", "energy"],
                   help="scalar used to colour the phase diagram")
    p.add_argument("--figure", default=None, help="output figure path (png)")

    # flow control
    p.add_argument("--dry-run", action="store_true",
                   help="print the commands without running or plotting")
    p.add_argument("--plot-only", action="store_true",
                   help="skip diagonalisation, just read existing output+plot")
    p.add_argument("--force", action="store_true",
                   help="re-run points even if their output already exists")
    return p


def main():
    args = get_parser().parse_args()
    if args.figure is None:
        args.figure = os.path.join(
            args.outdir, f"phasedia_3d_Jpm={args.jpm}_B={args.mag}_{args.observable}.png")

    if not os.path.isfile(args.lattice):
        sys.exit(f"lattice file not found: {args.lattice}")
    if not (args.plot_only or args.dry_run) and not os.path.isfile(args.binary):
        sys.exit(f"binary not found: {args.binary}\n(build it, or pass --binary)")

    directions = sample_triangle_directions(args.n_div)

    mag = args.mag

    # Assemble the full (shell x direction) list of field vectors.
    field_points = []
    for n_hat in directions:
        field_points.append((mag * n_hat, mag, n_hat))

    print(f"Lattice : {args.lattice}")
    print(f"J_pm    : {args.jpm}")
    print(f"B       : {args.mag}")
    print(f"Grid    : {len(directions)} directions "
          f" = {len(field_points)} runs")
    print(f"Output  : {args.outdir}\n")

    if not args.plot_only:
        run_grid(args, field_points)

    if args.dry_run:
        return

    rows = collect_results(args, field_points)
    plot_phase_diagram(args, rows)


if __name__ == "__main__":
    main()
