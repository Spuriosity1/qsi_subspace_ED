import csv
import sys
from os import path


in_csv = sys.argv[1]
out_planfile = path.join(sys.argv[2],'basis.plan')

if not in_csv.endswith('.csv'):
    print(f"USAGE: {sys.argv[0]} lat_manifest.csv out_path")


MAX_ST_SIZE = 17

inf = open(in_csv, 'r')
of = open(out_planfile, 'w')
reader = csv.reader(inf, delimiter=' ')
next(reader, None)

for line in reader:
    print(line)
    if int(line[1][:-1]) < MAX_ST_SIZE:
        of.write(f"../bin/gen_spinon_basis {line[0]}\n")

inf.close()
