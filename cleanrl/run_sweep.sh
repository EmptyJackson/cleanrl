#!/bin/bash
command=$(awk "NR==${SLURM_ARRAY_TASK_ID}" dohare_atari_sweep.txt)
echo "$command"
eval "$command" 