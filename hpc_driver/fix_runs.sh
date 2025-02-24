#!/bin/bash
if [ ! -e "./.git" ]; then
	echo "Run this script from the project directory"
	exit 1
fi

# Check if the user provided an input file
if [ "$#" -ne 3 ]; then
	echo "Usage: $0 <database> <latfile> <plan_file.plan>"
    exit 1
fi

database="$1"
latfile="$2"
planfile="$3"


MISSING_FILE=$(mktemp)

python3 scripts/check_completion.py $database $latfile --output_file $MISSING_FILE

i=1
rerun_planfile="$planfile.rerun$i"

while [ -f $rerun_planfile ]; do
	let i+=1
	rerun_planfile="$planfile.rerun$i"
done

for sector in `cat $MISSING_FILE`; do
	grep "${sector}" $planfile >> $rerun_planfile
done

echo "Written new plan to ${rerun_planfile}"
