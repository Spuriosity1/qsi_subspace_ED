import numpy as np
from ringflip_hamiltonian import RingflipHamiltonian
import pyrochlore
import argparse
from tqdm import tqdm
import datetime
TIMESTAMP = datetime.datetime.now(datetime.timezone.utc).isoformat()
# Generates NREPS rendom geometries and exports the lattice files.





rng = np.random.default_rng()


def random_quaternion():
    z = 2
    x = 0
    y = 0
    u = 0
    v = 0
    while z > 1:
        x = rng.uniform(-1, 1)
        y = rng.uniform(-1, 1)
        z = x*x + y*y
    w = 2
    while w > 1:
        u = rng.uniform(-1, 1)
        v = rng.uniform(-1, 1)
        w = u*u + v*v

    s = np.sqrt((1-z)/w)

    u *= s
    v *= s

    return np.array([[1-2*y**2-2*u**2, 2*x*y-2*u*v, 2*x*u+2*y*v],
                     [2*x*y+2*u*v, 1-2*x**2-2*u**2, 2*y*u-2*x*v],
                     [2*x*u-2*y*v, 2*y*u+2*x*v, 1-2*x**2-2*y**2]
                     ])


def get_random_RFH(max_size: int):
    # The task: Sample random integer-valued matrices somewhat uniformly on GL(3) subject to the constraints
    # -> no repeated indices, and
    # -> det R <= 24
    # idea 1: Generate random Haar SO(3), rescale by alpha, round to nearest int
    # -> unlikely to work, rounding will cause large errors
    # idea 2: exhaustion, sample uniformly on Z^3 (truncated to each individual entry < 24)
    # -> less controlled, but none of this is controlled anyway
    res = None
    while res is None:
        # spec = rng.integers(low=-12, high=12, size=(3,3))

        spec = (random_quaternion()*(max_size**0.333+0.5)).astype(np.int64)
        if np.abs(np.linalg.det(spec)) > max_size:
            continue
        res = RingflipHamiltonian(spec)
        if not res.rings_all_separate():
            res = None
    return res

if __name__ =="__main__":

    exported_files = {}
    
    ap = argparse.ArgumentParser()
    ap.add_argument("NREPS", type=int)
    ap.add_argument("--max_size", type=int, default=24)
    a = ap.parse_args()
    
    
    for i in tqdm(range(a.NREPS)):
        res = get_random_RFH(a.max_size)
        pyrochlore.export_json(res.lattice, res.latfile_loc)
        if res.latfile_loc in exported_files:
            exported_files[res.latfile_loc]['N'] += 1
        else:
            exported_files[res.latfile_loc] = {'N': 1, 'size': res.lattice.periodicity}
    
    
    with open(TIMESTAMP+"lattice_manifest.csv", 'w') as manifest:
        manifest.write("filename, repetitions, num_primitive\n")
        for f in exported_files:
            r = exported_files[f]
            manifest.write(f"{f} {r['N']}, {r['size'][0]*r['size'][1]*r['size'][2]}\n")
    

