import h5py
import sys

items = {}
def visit(name, o : h5py.Dataset):
    if o.size in items:
        items[o.size] += [name]
    else:
        items[o.size] = [name]

with h5py.File(sys.argv[1]) as f:
    f.visititems(visit)

for s in sorted(items.keys()):  
    #print(f"{s:8d} {items[s]}")
    print(' '.join(items[s][0].split('s')[-1].split('.')))
