#!/bin/bash

lf="$1" # The lattice file

# Directory where this script resides
scriptdir="$(realpath "$(dirname "$0")")"

stem=`basename ${1%.json}`
planfile="$scriptdir/phases_${stem}.plan"
odir="$HOME/out/phasedia/$stem"
mkdir -p $odir

echo "Writing plan file $planfile"
echo "Output directed to to $odir"


# Check that $lf exists and is a JSON file
if [[ ! -f "$lf" || "$lf" != *.json ]]; then
    echo "Error: input lattice file must exist and have a .json extension"
    exit 1
fi

rm -if "$planfile"

for jpm in $(seq -0.1 0.01 0.1); do
    for B in $(seq 0 0.005 0.1); do
        echo "$scriptdir/../build/diag_DOQSI_ham $lf --Jpm $jpm --B $B $B 0 -o $odir" -k 30 >> "$planfile"
        echo "$scriptdir/../build/diag_DOQSI_ham $lf --Jpm $jpm --B $B 0 0 -o $odir" -k 30 >> "$planfile"
    done
done

echo "wrote to ${planfile}"
