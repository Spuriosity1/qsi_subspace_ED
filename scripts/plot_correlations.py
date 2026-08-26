#!/usr/bin/env python3
"""Visualise the <O>, <OO> and <OOO> ring correlations produced by
diag_DOQSI_fused_mpi, grouped by plaquette sublattice.

The fused binary writes one HDF5 (`*.eigs.out.h5`) per (lattice, field) point,
containing (see src/diag_DOQSI_fused_mpi.cpp, geometry.hpp):

    ring            (n_ring, 1, 1)   <O_j>            one per 6-member ring
    ring_2          (n_ring, 1, 1)   <O_0' O_j>       the <OO> correlator
    flippability    (n_ring, 1, 1)   <O_j' O_j>       (not plotted here)
    partial_vol_slX (n_vol,  1, 1)   <O O O>          X = missing plaquette
                                                       sublattice (0..3)

The ring datasets are ordered exactly as the 6-member rings in the lattice
JSON's "rings" list, so ring j carries sublattice jdata["rings"][j]["sl"]
(0..3). The four partial_vol_slX datasets are already split by the sublattice
of the plaquette that was left out of the volume, so <OOO> is grouped by that.

Usage:
    python plot_correlations.py [OUT_DIR] [--latdir DIR] [--save DIR] [--no-show]

OUT_DIR defaults to /mnt/otus/final_out and is scanned recursively for
*.out.h5 files. Each file's lattice JSON is looked up by basename under
--latdir (default ../in relative to the repo root).
"""
import argparse
import glob
import json
import os

import h5py
import numpy as np
import matplotlib.pyplot as plt

# consistent colour per sublattice across every panel
SL_COLORS = {0: "tab:blue", 1: "tab:orange", 2: "tab:green", 3: "tab:red"}


def load_ring_geometry(latfile):
    """Return (sublattice, position, supercell_vectors) for the 6-member rings.

    `sl` is (n_ring,) in dataset order, `xyz` is (n_ring, 3), and `A` is the
    (3, 3) matrix of supercell lattice vectors (rows A0, A1, A2) used for the
    minimum-image convention.
    """
    with open(latfile) as fh:
        jdata = json.load(fh)
    rings = [r for r in jdata["rings"] if len(r["member_spin_idx"]) == 6]
    sl = np.array([r["sl"] for r in rings], dtype=int)
    xyz = np.array([r["xyz"] for r in rings], dtype=float)
    A = np.array([jdata["lattice_vectors"][k] for k in ("A0", "A1", "A2")],
                 dtype=float)
    return sl, xyz, A


def min_image_distances(xyz, A, ref=0):
    """Minimum-image distance from ring `ref` to every ring, under the
    periodic supercell spanned by the rows of `A`."""
    # candidate lattice translations n0 A0 + n1 A1 + n2 A2, |n_i| <= 2 covers
    # the nearest images for these compact cells
    rng = range(-2, 3)
    shifts = np.array([[i, j, k] for i in rng for j in rng for k in rng]) @ A
    dr = xyz - xyz[ref]                      # (n_ring, 3)
    cand = dr[:, None, :] - shifts[None, :, :]   # (n_ring, n_shift, 3)
    return np.sqrt((cand ** 2).sum(-1)).min(axis=1)


def find_latfile(out_h5, latdir):
    """Resolve the lattice JSON for an output file via its stored path.

    The `latfile_json` dataset holds the remote path used at compute time; we
    only trust its basename and look it up locally under `latdir`.
    """
    print(out_h5)
    with h5py.File(out_h5, "r") as f:
        remote = f["latfile_json"][()]
    remote = remote.decode() if isinstance(remote, bytes) else remote
    local = os.path.join(latdir, os.path.basename(remote))
    return local if os.path.exists(local) else None


def strip(ax, groups, title, ylabel):
    """One panel: values as a jittered strip, one column per sublattice."""
    rng = np.random.default_rng(0)
    for sl in sorted(groups):
        vals = np.asarray(groups[sl], dtype=float)
        if vals.size == 0:
            continue
        x = sl + (rng.random(vals.size) - 0.5) * 0.5
        ax.scatter(x, vals, s=18, alpha=0.7, color=SL_COLORS.get(sl, "gray"))
        ax.hlines(vals.mean(), sl - 0.35, sl + 0.35,
                  color="k", lw=2, zorder=3)
    ax.set_xticks(sorted(groups))
    ax.set_xlabel("plaquette sublattice")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim([-0.25,0.25])
    ax.grid(True, axis="y", alpha=0.3)


def oo_vs_distance(ax, OO, dist, ring_sl, ref_sl):
    """<O_0' O_j> against ring-ring distance, one series per ring-j sublattice.

    Points at the same distance are jittered a little so overlapping
    sublattices stay visible; the black bars mark the per-(distance,sl) mean.
    """
    rng = np.random.default_rng(0)
    uniq = np.unique(np.round(dist, 4))
    spacing = np.min(np.diff(uniq)) if uniq.size > 1 else 1.0
    for sl in np.unique(ring_sl):
        m = ring_sl == sl
        x = dist[m] + (rng.random(m.sum()) - 0.5) * 0.12 * spacing
        ax.scatter(x, OO[m], s=22, alpha=0.75, color=SL_COLORS.get(sl, "gray"),
                   label=f"ring$_j$ sl={sl}")
    # per (distance, sublattice) mean marker
    for d in uniq:
        for sl in np.unique(ring_sl):
            m = (np.round(dist, 4) == d) & (ring_sl == sl)
            if m.any():
                ax.hlines(OO[m].mean(), d - 0.18 * spacing, d + 0.18 * spacing,
                          color="k", lw=1.5, zorder=3)
    ax.set_xlabel(r"ring-ring distance $|r_j - r_0|$")
    ax.set_ylabel(r"$\langle OO \rangle$")
    ax.set_title(rf"$\langle O_0^\dagger O_j \rangle$ vs distance "
                 rf"(ring$_0$ sl={ref_sl})")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)


def plot_file(out_h5, latfile):
    ring_sl, xyz, A = load_ring_geometry(latfile)
    dist = min_image_distances(xyz, A, ref=0)
    with h5py.File(out_h5, "r") as f:
        O = np.ravel(f["ring"][()])          # <O_j>
        OO = np.ravel(f["ring_2"][()])       # <O_0' O_j>
        eig = float(np.ravel(f["eigenvalues"][()])[0])
        sector = f["sector"][()]
        sector = sector.decode() if isinstance(sector, bytes) else sector
        # <OOO>: one dataset per missing-plaquette sublattice
        ooo = {sl: np.ravel(f[f"partial_vol_sl{sl}"][()])
               for sl in range(4) if f"partial_vol_sl{sl}" in f}

    assert O.size == ring_sl.size, (
        f"ring count {O.size} != rings in JSON {ring_sl.size}")

    O_groups = {sl: O[ring_sl == sl] for sl in np.unique(ring_sl)}

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    strip(axes[0], O_groups, r"$\langle O \rangle$", r"$\langle O \rangle$")
    oo_vs_distance(axes[1], OO, dist, ring_sl, ring_sl[0])
    strip(axes[2], ooo,
          r"$\langle OOO \rangle$ (by missing plaq.)", r"$\langle OOO \rangle$")

    rel = os.path.basename(out_h5)
    fig.suptitle(f"{rel}\nsector={sector}   E0={eig:.6g}")
    fig.tight_layout()
    return fig


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", default="/mnt/otus/final_out",
                    help="directory scanned recursively for *.out.h5")
    ap.add_argument("--latdir", default=os.path.join(os.path.dirname(repo_root), "in"),
                    help="directory holding the lattice JSON files")
    ap.add_argument("--save", default=None,
                    help="directory to save PNGs into (created if needed)")
    ap.add_argument("--no-show", action="store_true",
                    help="do not open interactive windows")
    args = ap.parse_args()

    print(args.files)
    files = args.files

    if args.save:
        os.makedirs(args.save, exist_ok=True)

    for out_h5 in files:
        print(out_h5)
        latfile = find_latfile(out_h5, args.latdir)
        if latfile is None:
            print(f"[skip] {out_h5}: lattice JSON not found under {args.latdir}")
            continue
        fig = plot_file(out_h5, latfile)
        if args.save:
            # flatten <parent>/<name> into a single png name
            tag = os.path.basename(out_h5)
            png = os.path.join(args.save, tag + ".png")
            fig.savefig(png, dpi=120)
            print(f"[saved] {png}")

    if not args.no_show and not args.save:
        plt.show()


if __name__ == "__main__":
    main()
