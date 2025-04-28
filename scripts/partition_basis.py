from ringflip_hamiltonian import RingflipHamiltonian, build_matrix, ring_exp_values, calc_polarisation
import scipy.sparse.linalg as sLA
import pyrochlore
import numpy as np
import sys
import os

assert len(sys.argv) > 1, "usage: partition_basis LATTICE_FILE_NAME.json"
assert '.git' in os.listdir('.')

latfile = sys.argv[1]
file_name = os.path.basename(latfile)

lat = pyrochlore.import_json(latfile)

latvecs = np.array(lat.lattice_vectors)

rfh = RingflipHamiltonian(lat)
print("Setting up basis...")
rfh.load_basis(latfile.replace('.json','.0.basis.csv'))

sector_dir = os.path.join("basis_partitions/",
                          file_name.rsplit( ".", 1 )[ 0 ])

os.mkdir(sector_dir)

sectors = {}
for b in rfh.basis:
    s = calc_polarisation(rfh.lat, b)
    if not (s in sectors):
        sectors[s] = []
    s.append(b)

for s in sectors:
    print("SECTOR: " + str(s))
    sec_file = os.path.join(sector_dir, "s%d.%d.%d.%d_.csv" % s)
    with open(sec_file, 'w') as of:
        for b in sectors[s]:
            of.write('0x%08x\n' % b)


