#!/bin/bash
command=$(awk "NR==${SLURM_ARRAY_TASK_ID}" vb_atari_10_run.txt)
echo "$command"
eval "$command" 