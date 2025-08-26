#!/bin/bash

lf="$1" # The lattice file

# Directory where this script resides
scriptdir="$(realpath "$(dirname "$0")")"

stem=`basename ${1%.json}`
planfile="$scriptdir/phases_${stem}.plan"


# Check that $lf exists and is a JSON file
if [[ ! -f "$lf" || "$lf" != *.json ]]; then
    echo "Error: input lattice file must exist and have a .json extension"
    exit 1
fi

rm -if "$planfile"

for jpm in $(seq 0 0.002 0.05); do
    for B in $(seq 0 0.01 0.3); do
        echo "$scriptdir/build/diag_DOQSI_ham $lf --Jpm $jpm --B $B $B 0 -o $scriptdir/../out/phase_dia_B/222" >> "$planfile"
    done
done

echo "wrote to ${planfile}"
