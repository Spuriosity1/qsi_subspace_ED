from ringflip_hamiltonian import calc_polarisation, load_basis_csv
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

print("Setting up basis...")
basis = load_basis_csv(latfile.replace('.json','.0.basis.csv'))

sectors = {}
for b in basis:
    pol = calc_polarisation(lat, b)
    if pol not in sectors:
        sectors[pol] = []

    sectors[pol].append(b)


sector_dir = os.path.join("basis_partitions/",
                          file_name.rsplit( ".", 1 )[ 0 ])

os.mkdir(sector_dir)

for sector, bset in sectors.items():
    print("SECTOR: " + str(sector))
    sec_file = os.path.join(sector_dir, "s%d.%d.%d.%d_.csv" % sector)
    with open(sec_file, 'w') as of:
        for b in bset:
            of.write('0x%032x\n' % b)


