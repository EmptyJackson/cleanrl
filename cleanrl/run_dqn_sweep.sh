#!/bin/bash
command=$(awk "NR==${SLURM_ARRAY_TASK_ID}" dqn_atari_adammr_run.txt)
echo "$command"
eval "$command" 