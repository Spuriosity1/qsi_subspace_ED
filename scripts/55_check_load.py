from ringflip_hamiltonian import RingflipHamiltonian
import argparse
import pyrochlore
import time

def get_parser():
    ap = argparse.ArgumentParser(prog="PHASE_DIA")
    ap.add_argument("lattice_file", type=str)
    return ap


if __name__ == "__main__":

    ap = get_parser()
    a = ap.parse_args()

    lat = pyrochlore.import_json(a.lattice_file)
    
    rfh_csv = RingflipHamiltonian(lat, pyrochlore.get_rings)

    rfh_h5 = RingflipHamiltonian(lat, pyrochlore.get_rings)


    t_csv = -time.process_time()
    rfh_csv.load_basis(a.lattice_file[:-5] + '.0.basis.csv')
    t_csv += time.process_time()

    t_h5 = -time.process_time()
    rfh_h5.load_basis(a.lattice_file[:-5] + '.0.basis.h5')
    t_h5 += time.process_time()


    assert len(rfh_csv.basis) == len(rfh_h5.basis)

    for (bc, b5) in zip(rfh_csv.basis, rfh_h5.basis):
        if bc != b5:
            print("Disagreement: ", bc, " != ", b5)


    print(f"CSV load: {t_csv}")
    print(f"HDF5 load: {t_h5}")
