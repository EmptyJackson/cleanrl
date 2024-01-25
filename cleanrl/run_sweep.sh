#!/bin/bash
command=$(awk "NR==${SLURM_ARRAY_TASK_ID}" atari_run.txt)
echo "$command"
eval "$command" 