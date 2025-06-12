from ringflip_hamiltonian import RingflipHamiltonian, calc_polarisation
import pyrochlore
import sys
import numpy as np
import argparse
import os
from phasedia_impl import calc_spectrum




def get_parser():
    ap = argparse.ArgumentParser(prog="PHASE_DIA")
    ap.add_argument("lattice_file", type=str)
    ap.add_argument('g', type=float, nargs=4)
    ap.add_argument("--basis_file", type=str, default=None)
    ap.add_argument("--outdir","-o", type=str, required=True,
                    help="Database to store results in")
    ap.add_argument("--krylov_dim", type=int, default=200)

    return ap


def assert_unif_sector(rfh):
    sector = calc_polarisation(rfh.lattice, rfh.basis[0])
    if not all(calc_polarisation(rfh.lattice, b) == sector for b in rfh.basis):
        print("WARN: basis is not polarization-split")
        sector = None
    print("SECTOR: " + str(sector))
    return sector



def format_g_string(g):
    return ";".join(f"g{i}={val:.2g}" for i, val in enumerate(g))


if __name__ == "__main__":

    ap = get_parser()
    a = ap.parse_args()

    lat = pyrochlore.import_json(a.lattice_file)


    rfh = RingflipHamiltonian(lat, pyrochlore.get_ringflips)
    print("Setting up basis...")
    if a.basis_file is None:
        bfile = a.lattice_file.replace('.json', '.0.basis.csv')
    else:
        bfile = a.basis_file
    rfh.load_basis(bfile)

    outfile = os.path.join(a.outdir,  os.path.basename(a.lattice_file.replace('.json', 
                                                                              ';'+format_g_string(a.g))))

    sector = assert_unif_sector(rfh)

    e,v = calc_spectrum(a.g, rfh)
    np.savez_compressed(outfile, e=e,psi=v,latfile=a.lattice_file)

