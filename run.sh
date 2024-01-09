#!/bin/bash
WANDB_API_KEY=$(cat $HOME/.oxwhirl_wandb_api_key)
# git pull

script_and_args="${@:2}"
if [ $1 == "all" ]; then
    gpus="0 1 2 3 4 5 6 7"
else
    gpus=$1
fi

for gpu in $gpus; do
    echo "Launching container jaxrl_$gpu on GPU $gpu"
    docker run \
        --gpus device=$gpu \
        -e WANDB_API_KEY=$WANDB_API_KEY \
        --name jax_atari_$gpu \
        --user $(id -u) \
        --rm \
	    -d \
        -t jax_atari \
        /bin/bash -c "$script_and_args"
done