#!/usr/bin/env python3
"""Scan the magnetic field B in 3D at fixed Jpm and plot the average ring
flippability as three orthogonal slices (Bx=0, By=0, Bz=0) in a Nature-style
3D figure.

The ED is done with the older, single-threaded serial pipeline (no MPI):

    diag_DOQSI_ham   -- ground state within the ice subspace, given a basis file
    eval_observables -- per-ring flippability <psi| O_j^dag O_j |psi>

producing an HDF5 file whose name encodes (Jpm, Bx, By, Bz) -- exactly the
format found in ``../out/222``.  We call the pipeline on a grid of B vectors
lying on the three coordinate planes, read back the ``flippability`` dataset
(one entry per ring), average over rings, and render the slices.

Because each run is a plain single-threaded process, many field points can be
computed concurrently as ordinary subprocesses (``--jobs``) without any MPI
oversubscription; ``OMP_NUM_THREADS`` is pinned per process so the aggregate
thread count stays bounded.

Runs are cached: if the expected ``.eigs.out.h5`` already exists it is reused,
so the scan can be resumed / refined incrementally, and existing data is picked
up for free when grid points coincide.

Example
-------
    python scripts/ring_flippability_3d.py \
        --outdir ../out/ringflip_3d --Bmax 0.3 --N 21 \
        --jobs 8 --fig figures/ringflip_3d.png
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import h5py
import numpy as np


# --------------------------------------------------------------------------- #
#  Locating the binaries / lattice
# --------------------------------------------------------------------------- #
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(REPO, "..", "bin")
BUILD = os.path.join(REPO, "build", "src")


def _first_existing(*paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return paths[0]


DEFAULT_DIAG = _first_existing(os.path.join(BIN, "diag_DOQSI_ham"),
                               os.path.join(BUILD, "diag_DOQSI_ham"))
DEFAULT_EVAL = _first_existing(os.path.join(BIN, "eval_observables"),
                               os.path.join(BUILD, "eval_observables"))
DEFAULT_LAT = os.path.join(
    REPO, "test", "lattice_files", "pyro_2,0,0_0,2,0_0,0,2.json"
)
DEFAULT_BASIS = os.path.join(
    REPO, "..", "in", "pyro_2,0,0_0,2,0_0,0,2.0.basis.h5"
)


def base_name(Jpm: float, B) -> str:
    """Reproduce exactly the stem the serial binary writes
    (src/diag_DOQSI_ham.cpp:129): 4-decimal fixed-point fields.  ``-0.0`` is
    normalised to ``0.0`` so the grid centre matches the C ``%.4f`` of +0."""
    bx, by, bz = (0.0 if v == 0 else v for v in B)
    return f"Jpm={Jpm:.4f}%Bx={bx:.4f}%By={by:.4f}%Bz={bz:.4f}%"


# --------------------------------------------------------------------------- #
#  Running the scan (serial diag_DOQSI_ham -> eval_observables)
# --------------------------------------------------------------------------- #
@dataclass
class RunConfig:
    diag_bin: str
    eval_bin: str
    lattice: str
    basis_file: str
    latfile_dir: str
    Jpm: float
    outdir: str
    omp_threads: int
    extra: list  # passthrough args to diag_DOQSI_ham (e.g. --ncv, --algorithm)


def _run(cmd, env, cwd=None):
    return subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=cwd)


def run_point(cfg: RunConfig, B) -> str:
    """Ensure the flippability result for field ``B`` exists on disk, running
    the two-step serial pipeline if it is not already cached.  Returns the
    ``.eigs.out.h5`` path."""
    stem = base_name(cfg.Jpm, B)
    eigs = os.path.join(cfg.outdir, stem + ".eigs.h5")
    out = os.path.join(cfg.outdir, stem + ".eigs.out.h5")
    if os.path.exists(out):
        return out

    # Single-threaded processes: cap OpenMP so N concurrent jobs use ~N cores.
    env = dict(os.environ, OMP_NUM_THREADS=str(cfg.omp_threads))

    # Step 1: diagonalise -> eigenvector(s) in {stem}.eigs.h5
    if not os.path.exists(eigs):
        diag_cmd = [
            cfg.diag_bin, cfg.lattice,
            "--Jpm", f"{cfg.Jpm}",
            "--B", f"{B[0]}", f"{B[1]}", f"{B[2]}",
            "--basis_file", cfg.basis_file,
            "--output_dir", cfg.outdir,
        ] + cfg.extra
        p = _run(diag_cmd, env)
        if p.returncode != 0 or not os.path.exists(eigs):
            sys.stderr.write(
                f"\n[FAIL diag] B={B}\n  cmd: {' '.join(diag_cmd)}\n"
                f"  stderr tail:\n{p.stderr[-800:]}\n")
            raise RuntimeError(f"diag_DOQSI_ham failed for B={B}")

    # Step 2: evaluate observables -> {stem}.eigs.out.h5 (ring <O> + flippability)
    eval_cmd = [
        cfg.eval_bin, eigs,
        "--latfile_dir", cfg.latfile_dir,
        "--calculate", "ring", "flippability",
    ]
    p = _run(eval_cmd, env)
    if p.returncode != 0 or not os.path.exists(out):
        sys.stderr.write(
            f"\n[FAIL eval] B={B}\n  cmd: {' '.join(eval_cmd)}\n"
            f"  stderr tail:\n{p.stderr[-800:]}\n")
        raise RuntimeError(f"eval_observables failed for B={B}")
    return out


def avg_flippability(path: str) -> float:
    """Average of the per-ring flippability <psi|O_j^dag O_j|psi>."""
    with h5py.File(path, "r") as f:
        return float(np.mean(np.asarray(f["flippability"]).ravel()))


# --------------------------------------------------------------------------- #
#  Grid construction: the three coordinate planes bounding the positive octant
# --------------------------------------------------------------------------- #
def plane_grids(Bmax: float, N: int):
    """Return, for each of the three coordinate slice planes (Bz=0, By=0, Bx=0),
    a dict with the two in-plane axis values and the 3D field vectors on that
    plane.  We sample only the positive octant (Bx,By,Bz >= 0); the other seven
    octants are related by the cubic point-group symmetry of the field, so the
    three quarter-planes meeting at the origin carry all the information."""
    ax = np.linspace(0.0, Bmax, N)
    U, V = np.meshgrid(ax, ax, indexing="ij")
    planes = {
        # name -> (fixed component index, builder that maps (u,v)->(Bx,By,Bz))
        "Bz=0": (2, lambda u, v: (u, v, 0.0)),   # (Bx, By) plane
        "By=0": (1, lambda u, v: (u, 0.0, v)),   # (Bx, Bz) plane
        "Bx=0": (0, lambda u, v: (0.0, u, v)),   # (By, Bz) plane
    }
    out = {}
    for name, (_, build) in planes.items():
        B = np.empty((N, N, 3))
        for i in range(N):
            for j in range(N):
                B[i, j] = build(U[i, j], V[i, j])
        out[name] = dict(axis=ax, U=U, V=V, B=B)
    return out


# --------------------------------------------------------------------------- #
#  Plotting
# --------------------------------------------------------------------------- #
def set_nature_style():
    import matplotlib as mpl
    mpl.use("Agg")  # headless: we only ever save to file
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "figure.dpi": 150,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "mathtext.fontset": "dejavusans",
    })


def plot_slices(planes, values, Jpm, Bmax, cmap_name, fig_path):
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors, colormaps

    set_nature_style()

    allv = np.concatenate([values[n].ravel() for n in values])
    norm = colors.Normalize(vmin=np.nanmin(allv), vmax=np.nanmax(allv))
    cmap = colormaps[cmap_name]

    fig = plt.figure(figsize=(6.4, 5.2))
    ax = fig.add_subplot(111, projection="3d")

    # map each plane's (U, V) into 3D (X, Y, Z)
    to_xyz = {
        "Bz=0": lambda U, V: (U, V, np.zeros_like(U)),
        "By=0": lambda U, V: (U, np.zeros_like(U), V),
        "Bx=0": lambda U, V: (np.zeros_like(U), U, V),
    }

    for name in ("Bz=0", "By=0", "Bx=0"):
        p = planes[name]
        X, Y, Z = to_xyz[name](p["U"], p["V"])
        fc = cmap(norm(values[name]))
        ax.plot_surface(
            X, Y, Z,
            facecolors=fc,
            rstride=1, cstride=1,
            linewidth=0, antialiased=False,
            shade=False, zorder=1,
        )

    lim = Bmax * 1.02
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_zlim(0, lim)
    ax.set_xlabel(r"$B_x\,/\,J_{zz}$", labelpad=8)
    ax.set_ylabel(r"$B_y\,/\,J_{zz}$", labelpad=8)
    ax.set_zlabel(r"$B_z\,/\,J_{zz}$", labelpad=6)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=24, azim=45)

    # de-clutter: light panes, thin grid (the "Nature" look). API for pane /
    # grid styling has churned across matplotlib versions, so guard it.
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        try:
            a.set_pane_color((1, 1, 1, 0))
        except Exception:
            pass
        try:
            a._axinfo["grid"].update(color=(0.85, 0.85, 0.85), linewidth=0.4)
        except Exception:
            pass
    ax.tick_params(pad=1)

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, shrink=0.6, aspect=18, pad=0.10)
    cb.set_label(r"average ring flippability $\overline{\langle O^\dagger O\rangle}$")
    cb.outline.set_linewidth(0.5)

    ax.set_title(
        rf"Average ring flippability, $J_\pm = {Jpm:.3f}\,J_{{zz}}$",
        pad=2,
    )

    fig.savefig(fig_path)
    root, _ = os.path.splitext(fig_path)
    fig.savefig(root + ".pdf")
    print(f"[fig] wrote {fig_path} and {root}.pdf")


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lattice", default=DEFAULT_LAT, help="lattice JSON")
    ap.add_argument("--diag-bin", default=DEFAULT_DIAG,
                    help="serial diag_DOQSI_ham binary")
    ap.add_argument("--eval-bin", default=DEFAULT_EVAL,
                    help="serial eval_observables binary")
    ap.add_argument("--basis-file", default=DEFAULT_BASIS,
                    help="pre-generated on-disk ice-state basis (HDF5)")
    ap.add_argument("--Jpm", type=float, default=-0.050)
    ap.add_argument("--Bmax", type=float, default=0.30,
                    help="half-width of the field scan in each axis (units Jzz)")
    ap.add_argument("--N", type=int, default=21,
                    help="grid points per axis per plane (odd -> includes B=0)")
    ap.add_argument("--outdir", default=os.path.join(REPO, "..", "out", "ringflip_3d"),
                    help="HDF5 cache/output dir for the ED runs")
    ap.add_argument("--jobs", type=int, default=8,
                    help="concurrent (single-threaded) pipeline invocations")
    ap.add_argument("--omp-threads", type=int, default=1,
                    help="OMP_NUM_THREADS per process (keep low when jobs is high)")
    ap.add_argument("--cmap", default="magma")
    ap.add_argument("--fig", default=os.path.join(REPO, "scripts", "figures",
                                                  "ringflip_3d.png"))
    ap.add_argument("--dry-run", action="store_true",
                    help="build grid and report work, but do not run the binary")
    ap.add_argument("extra", nargs=argparse.REMAINDER,
                    help="extra args forwarded to diag_DOQSI_ham (after --)")
    args = ap.parse_args()

    if args.N % 2 == 0:
        print("[warn] N is even -> B=0 not sampled exactly; consider an odd N.")

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.fig)), exist_ok=True)

    extra = args.extra
    if extra and extra[0] == "--":
        extra = extra[1:]

    lattice = os.path.abspath(args.lattice)
    cfg = RunConfig(
        diag_bin=os.path.abspath(args.diag_bin),
        eval_bin=os.path.abspath(args.eval_bin),
        lattice=lattice,
        basis_file=os.path.abspath(args.basis_file),
        latfile_dir=os.path.dirname(lattice),
        Jpm=args.Jpm, outdir=os.path.abspath(args.outdir),
        omp_threads=args.omp_threads, extra=extra,
    )

    # eval_observables locates the basis as {latfile_dir}/{stem}.0.basis.h5.
    # Make sure that path resolves to the requested basis file.
    expected_basis = os.path.join(
        cfg.latfile_dir,
        os.path.basename(lattice).replace(".json", ".0.basis.h5"))
    if not os.path.exists(expected_basis):
        try:
            os.symlink(cfg.basis_file, expected_basis)
            print(f"[basis] linked {expected_basis} -> {cfg.basis_file}")
        except OSError as e:
            print(f"[warn] could not link basis next to lattice: {e}")

    planes = plane_grids(args.Bmax, args.N)

    def out_path(B):
        return os.path.join(cfg.outdir, base_name(cfg.Jpm, B) + ".eigs.out.h5")

    # unique B vectors across all three planes (share axes + origin)
    uniq = {}
    for name, p in planes.items():
        for i in range(args.N):
            for j in range(args.N):
                B = tuple(round(float(x), 4) for x in p["B"][i, j])
                uniq.setdefault(B, None)
    todo = [B for B in uniq if not os.path.exists(out_path(B))]
    print(f"[grid] {len(uniq)} unique field points "
          f"({len(uniq) - len(todo)} cached, {len(todo)} to compute)")

    if args.dry_run:
        return

    # --- run the scan (cached, parallel) ---
    done = 0
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(run_point, cfg, B): B for B in uniq}
        for fut in futs:
            fut.result()
            done += 1
            if done % 25 == 0 or done == len(futs):
                print(f"  [{done}/{len(futs)}] runs ready", flush=True)

    # --- assemble slice fields ---
    values = {}
    for name, p in planes.items():
        V = np.empty((args.N, args.N))
        for i in range(args.N):
            for j in range(args.N):
                B = tuple(round(float(x), 4) for x in p["B"][i, j])
                V[i, j] = avg_flippability(out_path(B))
        values[name] = V

    plot_slices(planes, values, args.Jpm, args.Bmax, args.cmap, args.fig)


if __name__ == "__main__":
    main()
