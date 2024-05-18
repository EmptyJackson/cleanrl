FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# install ubuntu dependencies
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get -y install python3-pip xvfb ffmpeg git build-essential
RUN ln -s /usr/bin/python3 /usr/bin/python

# install python dependencies
RUN mkdir cleanrl_utils && touch cleanrl_utils/__init__.py
COPY ./requirements/ /requirements
# RUN pip install -r /requirements/requirements-envpool.txt
RUN pip install -r /requirements/requirements-atari.txt
RUN pip install -r /requirements/requirements-jax.txt
RUN pip install --upgrade "jax[cuda11_pip]==0.4.7" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN pip install mle_logging

# install mujoco_py
RUN apt-get -y install wget unzip software-properties-common \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev patchelf
# RUN poetry install -E "atari mujoco_py"
# RUN poetry run python -c "import mujoco_py"

COPY entrypoint.sh /usr/local/bin/
# RUN chmod 777 /usr/local/bin/entrypoint.sh
# ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# copy local files
COPY ./cleanrl /cleanrl
WORKDIR /cleanrl
