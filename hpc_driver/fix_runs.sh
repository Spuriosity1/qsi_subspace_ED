#!/bin/bash
if [ ! -e "./.git" ]; then
	echo "Run this script from the project directory"
	exit 1
fi

# Check if the user provided an input file
if [ "$#" -lt 3 ]; then
	echo "Usage: $0 <database> <latfile> <plan_file.plan> <rotation=I,X,Y,Z>"
    exit 1
fi

database="$1"
latfile="$2"
planfile="$3"
if [ "$#" -ge 4 ]; then
	rot="$4"
else
	rot="I"
fi

MISSING_FILE=$(mktemp)

python3 scripts/check_completion.py $database $latfile --output_file $MISSING_FILE --rotation $rot

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
