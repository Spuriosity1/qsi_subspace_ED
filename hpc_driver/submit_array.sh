#!/bin/bash

# Check if the user provided an input file
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <command_file> <cpus_per_task>"
    exit 1
fi

COMMAND_FILE="$1"
CPUS_PER_TASK="$2"

# Check if the file exists
if [ ! -f "$COMMAND_FILE" ]; then
    echo "Error: File '$COMMAND_FILE' not found."
    exit 1
fi


# Determine the template file based on the first line of the command file
FIRST_LINE=$(head -n 1 "$COMMAND_FILE")
if [[ "$FIRST_LINE" == python* ]]; then
    TEMPLATE_FILE="pyjob_template.slurm"
else
    TEMPLATE_FILE="cppjob_template.slurm"
fi

if [ ! -f "$TEMPLATE_FILE" ]; then
    echo "Error: File '$TEMPLATE_FILE' not found."
    exit 1
fi


# Get the number of lines (commands) in the file
NUM_LINES=$(wc -l < "$COMMAND_FILE")

# Create a temporary job script
JOB_SCRIPT=$(mktemp)


# Copy template content to job script
cp "$TEMPLATE_FILE" "$JOB_SCRIPT"

echo "
# Load the command from the file
COMMAND=\$(sed \"\${SLURM_ARRAY_TASK_ID}q;d\" \"$COMMAND_FILE\")

echo \"Running: \$COMMAND\"
eval \"\$COMMAND\"" >> "$JOB_SCRIPT"

# Submit the array job

sbatch --array=1-$NUM_LINES --cpus-per-task=$CPUS_PER_TASK "$JOB_SCRIPT"

echo "Submitted array job with $NUM_LINES tasks."
#rm "$JOB_SCRIPT"
echo Wrote job script $JOB_SCRIPT

