#!/usr/bin/env python3
"""Render the boundary surface(s) of the average ring flippability in 3D field
space, in the positive octant (Bx, By, Bz >= 0) at fixed Jpm.

The average ring flippability is effectively an indicator field: in some region
the ground state supports ring flips (flippability ~ 0.125) and elsewhere it is
a frozen Ising configuration (flippability ~ 0), separated by a sharp boundary.
That region is *not* simply a star about the origin -- it has disconnected
pieces and thin fins running along the high-symmetry ridges (the <110>-<111>
arcs), so a single radial r(theta, phi) cannot represent it.  Instead we use
marching cubes, which triangulates an isosurface of arbitrary topology:

  1. sample a *coarse* 3D indicator grid over the octant and label each node
     flippable / frozen;
  2. run marching cubes on the indicator to get the boundary triangulation --
     this captures every connected component and re-entrant fin correctly;
  3. each marching-cubes vertex lies on a coarse grid edge straddling the
     boundary; **bisect** that edge (Bolzano / intermediate-value: g(B) =
     flippability(B) - tol changes sign) to snap the vertex to the true boundary
     at the ED field resolution.

So the coarse grid fixes the topology and the bisection buys the accuracy, at
O(volume) coarse evals + O(surface area) refinement evals.

The ED reuses the single-threaded serial pipeline from ``ring_flippability_3d.py``
(diag_DOQSI_ham -> eval_observables), cached on disk.

Example
-------
    python scripts/flippability_boundary.py \
        --outdir ../out/ringflip_3d --Bmax 0.3 --n 41 --refine 6 \
        --jobs 12 --fig figures/flippability_boundary.png
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import h5py
import numpy as np

import ring_flippability_3d as rf  # shared serial ED runner + defaults

try:
    from tqdm import tqdm
except ImportError:                     # lightweight fallback
    tqdm = None


# The ED writes field components with 4-decimal precision -- the finest
# meaningful spacing; bisecting below it just re-requests the same computation.
BFMT = 1e-4


def _key(B):
    return tuple(round(float(x), 4) for x in B)


# --------------------------------------------------------------------------- #
#  Cached, parallel flippability evaluator
# --------------------------------------------------------------------------- #
class Evaluator:
    """Memoised average-flippability oracle backed by the serial ED pipeline."""

    def __init__(self, cfg: rf.RunConfig, jobs: int):
        self.cfg = cfg
        self.jobs = jobs
        self.cache: dict[tuple, float] = {}
        self.n_eval = 0

    def evaluate_many(self, points, desc="ED runs") -> None:
        """Populate the cache for every B in ``points`` (parallel, deduped),
        showing a live progress indicator."""
        need = {}
        for B in points:
            k = _key(B)
            if k not in self.cache:
                need[k] = B
        if not need:
            return
        total = len(need)
        with ThreadPoolExecutor(max_workers=self.jobs) as ex:
            futs = {ex.submit(self._one, B): k for k, B in need.items()}
            if tqdm is not None:
                for fut in tqdm(as_completed(futs), total=total, desc=desc,
                                unit="run", leave=False, dynamic_ncols=True):
                    self.cache[futs[fut]] = fut.result()
            else:
                done, t0, last = 0, time.time(), 0.0
                for fut in as_completed(futs):
                    self.cache[futs[fut]] = fut.result()
                    done += 1
                    now = time.time()
                    if now - last > 0.2 or done == total:
                        rate = done / max(now - t0, 1e-9)
                        sys.stderr.write(
                            f"\r  [{desc}] {done}/{total} "
                            f"({rate:5.1f} run/s)   ")
                        sys.stderr.flush()
                        last = now
                sys.stderr.write("\n")
        self.n_eval += total

    def _one(self, B) -> float:
        return rf.avg_flippability(rf.run_point(self.cfg, B))

    def value(self, B) -> float:
        return self.cache[_key(B)]


# --------------------------------------------------------------------------- #
#  Coarse indicator grid -> marching-cubes topology -> bisected vertices
# --------------------------------------------------------------------------- #
def indicator_grid(ev: Evaluator, Bmax: float, n: int, tol: float):
    """Evaluate the n^3 coarse grid over the positive octant and return the
    axis coordinates and a boolean 'flippable' array."""
    ax = np.linspace(0.0, Bmax, n)
    nodes = [(ax[i], ax[j], ax[k])
             for i in range(n) for j in range(n) for k in range(n)]
    ev.evaluate_many(nodes, desc="coarse grid")
    V = np.empty((n, n, n), dtype=bool)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                V[i, j, k] = ev.value((ax[i], ax[j], ax[k])) > tol
    return ax, V


def marching_surface(V, h):
    """Marching cubes on the (float) indicator at level 0.5.  Returns verts (in
    physical Bx,By,Bz coordinates) and integer triangle faces."""
    from skimage import measure
    verts, faces = measure.marching_cubes(
        V.astype(np.float32), level=0.5, spacing=(h, h, h))[:2]
    return verts, faces


def snap_vertices(ev: Evaluator, verts, ax, V, tol, refine):
    """Each marching-cubes vertex sits at the midpoint of a coarse grid edge
    that straddles the boundary.  Recover that edge, then bisect it (lock-step,
    in parallel) to move the vertex onto the true boundary."""
    n = len(ax)
    h = ax[1] - ax[0]
    t = verts / h                                  # fractional grid indices
    frac = np.abs(t - np.round(t))
    axis = np.argmax(frac, axis=1)                 # the interpolated axis
    base = np.clip(np.round(t).astype(int), 0, n - 1)

    lo = np.empty_like(verts)                      # flippable endpoints
    hi = np.empty_like(verts)                      # frozen endpoints
    for m in range(len(verts)):
        a = axis[m]
        k0 = int(np.clip(np.floor(t[m, a]), 0, n - 1))
        k1 = int(np.clip(k0 + 1, 0, n - 1))
        iA = base[m].copy(); iA[a] = k0
        iB = base[m].copy(); iB[a] = k1
        pA = ax[iA]
        pB = ax[iB]
        if V[tuple(iA)]:
            lo[m], hi[m] = pA, pB
        else:
            lo[m], hi[m] = pB, pA

    for d in range(refine):
        act = np.abs(hi - lo).sum(axis=1) > BFMT
        if not act.any():
            break
        mid = 0.5 * (lo + hi)
        ev.evaluate_many(mid[act], desc=f"snap {d + 1}/{refine}")
        for m in np.nonzero(act)[0]:
            if ev.value(mid[m]) > tol:
                lo[m] = mid[m]
            else:
                hi[m] = mid[m]
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------- #
#  Plot
# --------------------------------------------------------------------------- #
def plot_surface_mesh(verts, faces, Bmax, Jpm, cmap_name, fig_path):
    rf.set_nature_style()
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors, colormaps
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    tris = verts[faces]                                  # (F, 3, 3)
    centroids = tris.mean(axis=1)
    r = np.linalg.norm(centroids, axis=1)               # colour by |B|
    cmap = colormaps[cmap_name]
    norm = colors.Normalize(vmin=float(r.min()), vmax=float(r.max()))

    fig = plt.figure(figsize=(6.6, 5.4))
    ax = fig.add_subplot(111, projection="3d")

    coll = Poly3DCollection(tris, linewidths=0.1)
    coll.set_facecolor(cmap(norm(r)))
    coll.set_edgecolor((1, 1, 1, 0.12))
    ax.add_collection3d(coll)

    lim = Bmax * 1.02
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_zlim(0, lim)
    ax.set_xlabel(r"$B_x\,/\,J_{zz}$", labelpad=8)
    ax.set_ylabel(r"$B_y\,/\,J_{zz}$", labelpad=8)
    ax.set_zlabel(r"$B_z\,/\,J_{zz}$", labelpad=6)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=24, azim=45)

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
    cb.set_label(r"$|\mathbf{B}|\,/\,J_{zz}$ on the boundary")
    cb.outline.set_linewidth(0.5)

    ax.set_title(
        rf"Ring-flippability boundary surface, $J_\pm = {Jpm:.3f}\,J_{{zz}}$",
        pad=2)

    fig.savefig(fig_path)
    root, _ = os.path.splitext(fig_path)
    fig.savefig(root + ".pdf")
    print(f"[fig] wrote {fig_path} and {root}.pdf")


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lattice", default=rf.DEFAULT_LAT)
    ap.add_argument("--diag-bin", default=rf.DEFAULT_DIAG)
    ap.add_argument("--eval-bin", default=rf.DEFAULT_EVAL)
    ap.add_argument("--basis-file", default=rf.DEFAULT_BASIS)
    ap.add_argument("--Jpm", type=float, default=-0.050)
    ap.add_argument("--Bmax", type=float, default=0.30,
                    help="extent of the field scan in each axis (units Jzz)")
    ap.add_argument("--n", type=int, default=41,
                    help="coarse indicator grid points per axis (sets topology "
                         "resolution: thin fins need a finer grid)")
    ap.add_argument("--refine", type=int, default=6,
                    help="bisection depth used to snap each surface vertex")
    ap.add_argument("--tol", type=float, default=1e-6,
                    help="flippability threshold separating frozen (< tol) "
                         "from flippable (> tol)")
    ap.add_argument("--outdir",
                    default=os.path.join(rf.REPO, "..", "out", "ringflip_3d"))
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--omp-threads", type=int, default=1)
    ap.add_argument("--cmap", default="viridis")
    ap.add_argument("--fig", default=os.path.join(
        rf.REPO, "scripts", "figures", "flippability_boundary.png"))
    ap.add_argument("--save-mesh", default=None,
                    help="optional .npz path to dump (verts, faces)")
    ap.add_argument("extra", nargs=argparse.REMAINDER)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.fig)), exist_ok=True)

    extra = args.extra
    if extra and extra[0] == "--":
        extra = extra[1:]

    lattice = os.path.abspath(args.lattice)
    cfg = rf.RunConfig(
        diag_bin=os.path.abspath(args.diag_bin),
        eval_bin=os.path.abspath(args.eval_bin),
        lattice=lattice,
        basis_file=os.path.abspath(args.basis_file),
        latfile_dir=os.path.dirname(lattice),
        Jpm=args.Jpm, outdir=os.path.abspath(args.outdir),
        omp_threads=args.omp_threads, extra=extra,
    )

    # eval_observables locates the basis next to the lattice file.
    expected_basis = os.path.join(
        cfg.latfile_dir,
        os.path.basename(lattice).replace(".json", ".0.basis.h5"))
    if not os.path.exists(expected_basis):
        try:
            os.symlink(cfg.basis_file, expected_basis)
            print(f"[basis] linked {expected_basis} -> {cfg.basis_file}")
        except OSError as e:
            print(f"[warn] could not link basis next to lattice: {e}")

    ev = Evaluator(cfg, args.jobs)

    print(f"[grid] evaluating {args.n}^3 = {args.n**3} indicator nodes ...")
    ax, V = indicator_grid(ev, args.Bmax, args.n, args.tol)
    print(f"[grid] flippable nodes: {int(V.sum())} / {V.size}")
    if not V.any() or V.all():
        print("[!] indicator is uniform -- no boundary. Adjust --Bmax / --n.")
        return

    h = ax[1] - ax[0]
    verts, faces = marching_surface(V, h)
    print(f"[mc] {len(verts)} vertices, {len(faces)} triangles; "
          f"snapping vertices with depth-{args.refine} bisection ...")
    verts = snap_vertices(ev, verts, ax, V, args.tol, args.refine)
    verts = np.clip(verts, 0.0, args.Bmax)
    print(f"[done] {ev.n_eval} ED runs total")

    if args.save_mesh:
        np.savez(args.save_mesh, verts=verts, faces=faces)
        print(f"[mesh] wrote {args.save_mesh}")

    plot_surface_mesh(verts, faces, args.Bmax, args.Jpm, args.cmap, args.fig)


if __name__ == "__main__":
    main()
