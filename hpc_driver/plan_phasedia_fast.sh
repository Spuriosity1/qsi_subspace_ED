#!/bin/bash

# Check if the user provided an input file
if [ "$#" -lt 5 ]; then
    echo "Usage: $0 <latspec> <initial_index> <rotation=IXYZ> <name> <db_repo> < 0.1 0.2 ..."
    exit 1
fi

lattice=`basename $1`

sector_decomp="../basis_partitions/$lattice"

if [ ! -d "$sector_decomp" ]; then
	echo "Error: sector decomposition not found at $sector_decomp"
	exit 1
fi

sector_list=`ls $sector_decomp`;


latfile="../lattice_files/$lattice.json"
ROTATION="${3}"
NAME="${4}"

DB_REPO="${5}"

if [ ! -d "${DB_REPO}" ]; then
    echo "Error: specified db repo does not exist. Create?"
    read -p "Continue \(y/n\)?" choice
    case "$choice" in 
      y|Y ) echo "yes"; mkdir -p "${DB_REPO}";;
      n|N ) echo "no"; exit 1;;
      * ) echo "invalid"; exit 1;;
    esac
fi

echoerr() { echo "$@" 1>&2; }

index=$2

file1="${NAME}.plan_small"
file2="${NAME}.plan_large"

if [[ -f $file1 || -f $file2 ]]; then
	echoerr "ERROR! files $file1 $file2 exist"
	exit 1
fi

while read -r x; do
	echo "x =$x"
	for p in $sector_list; do
		bfile="../basis_partitions/$lattice/$p"
		cmd="python3 ../scripts/phasedia_micro.py $latfile $x --db_file ${DB_REPO}/data${NAME}_$index.db --basis_file $bfile --rotation $ROTATION"
		if [[ `wc -l $bfile | sed -re 's/([0-9]+).*/\1/'` -lt 100000 ]]; then
			echo $cmd >> $file1
		else
			echo $cmd >> $file2
		fi
		let index+=1
	done
done < /dev/stdin

echo "Wrote files $file1 $file2"
