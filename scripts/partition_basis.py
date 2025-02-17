from ringflip_hamiltonian import RingflipHamiltonian, build_matrix, ring_exp_values
import scipy.sparse.linalg as sLA
import pyrochlore
import numpy.linalg as LA
import numpy as np
from tqdm import tqdm
from db import connect_npsql
import sys
import os

assert len(sys.argv) > 1, "usage: partition_basis BASIS_FILE_NAME"
assert '.git' in os.listdir('.')

latfile = sys.argv[1]
file_name = os.path.basename(latfile)

lat = pyrochlore.import_json(latfile)


latvecs = np.array(lat.lattice_vectors)

rfh = RingflipHamiltonian(lat)
print("Setting up basis...")
rfh.calc_basis()

sector_dir = os.path.join("basis_partitions/",
                          file_name.rsplit( ".", 1 )[ 0 ])

os.mkdir(sector_dir)

for sector in rfh.sectors:
    print("SECTOR: " + str(sector))
    sec_file = os.path.join(sector_dir, "s%d.%d.%d.%d_" % sector)
    with open(sec_file, 'w') as of:
        for b in rfh.basis[sector]:
            of.write('0x%08x\n' % b)


