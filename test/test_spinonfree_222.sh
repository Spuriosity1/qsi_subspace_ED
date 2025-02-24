#!/bin/bash

data_dir="data"
stem="pyro_2_2_2x0,4,4b4,0,4b4,4,0b1"

infile="${data_dir}/${stem}.json"
gen_outfile="${data_dir}/${stem}.0.test_basis.csv"
ref_outfile="${data_dir}/${stem}.reference.basis.csv"

sort "${ref_outfile}" > "${ref_outfile}.sorted"
ref_outfile="${ref_outfile}.sorted"


CMD1="../build/gen_spinon_basis $infile 0 .test_basis"
echo $CMD1
eval "$CMD1" > tmp.txt

DCMD="sort ${gen_outfile} | diff - ${ref_outfile}"
echo $DCMD
eval "$DCMD"

if [[ $? -eq 0 ]]; then
	echo "single-thread test passed"
else
	echo "single-thread test failed!"
fi


CMD2="../build/gen_spinon_basis_parallel $infile 4 0 .test_basis"
echo $CMD2
eval $CMD2 > tmp2.txt

echo $DCMD
eval "$DCMD"

if [[ $? -eq 0 ]]; then
	echo "parallel test passed"
else
	echo "parallel test failed!"
fi

rm "${gen_outfile}"
