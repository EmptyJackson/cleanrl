#!/bin/bash
command=$(awk "NR==${SLURM_ARRAY_TASK_ID}" c51_atari_10_run.txt)
echo "$command"
eval "$command" 