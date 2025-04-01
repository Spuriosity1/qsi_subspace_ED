#!/bin/bash

mkdir -p "tmp"

data_dir="data"
stem="pyro_2_2_2x0,4,4b4,0,4b4,4,0b1"

infile="${data_dir}/${stem}.json"

ext_st=".test_basis_st"
ext_par=".test_basis_st"

gen_outfile_st="${data_dir}/${stem}.0${ext_st}.csv"
gen_outfile_par="${data_dir}/${stem}.0${ext_par}.csv"
ref_outfile="${data_dir}/${stem}.reference.basis.csv"

sort "${ref_outfile}" > "${ref_outfile}.sorted"
ref_outfile="${ref_outfile}.sorted"


CMD1="../build/gen_spinon_basis $infile 0 $ext_st"
echo $CMD1
eval "$CMD1" > tmp/output_st.txt

DCMD="sort ${gen_outfile_st} | diff - ${ref_outfile}"
echo $DCMD
eval "$DCMD"

if [[ $? -eq 0 ]]; then
	echo -e "\033[32;1;4msingle-threaded test passed!\033[0m"
else
	echo -e "\033[31;1;4msingle-threaded test failed!\033[0m"
fi


CMD2="../build/gen_spinon_basis_parallel $infile 4 0 $ext_par"
echo $CMD2
eval $CMD2 > tmp/output_par.txt


DCMD="sort ${gen_outfile_par} | diff - ${ref_outfile}"
echo $DCMD
eval "$DCMD"

if [[ $? -eq 0 ]]; then
	echo -e "\033[32;1;4mparallel test passed!\033[0m"
else
	echo -e "\033[31;1;4mparallel test failed!\033[0m"
fi

rm "${gen_outfile}"
