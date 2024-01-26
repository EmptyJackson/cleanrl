#!/bin/bash
command=$(awk "NR==${SLURM_ARRAY_TASK_ID}" count_reset_atari_run.txt)
echo "$command"
eval "$command" 