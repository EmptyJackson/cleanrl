#!/bin/bash


docker build \
    --build-arg ENV_REQS="$(cat requirements/requirements-envpool.txt | tr '\n' ' ')" \
    --build-arg JAX_REQS="$(cat requirements/requirements-jax.txt | tr '\n' ' ')" \
    -t jax_atari \
    .
