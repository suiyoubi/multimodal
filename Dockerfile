# NVIDIA CUDA image as a base
#FROM nvidia/cuda:11.6.0-base-ubuntu20.04
FROM  nvcr.io/nvidia/pytorch:22.02-py3

# Remove the old GPG key and set up the new one
#RUN apt install curl # REQUIRED apt-get update FIRST
#RUN curl -O https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
#RUN dpkg -i cuda-keyring_1.0-1_all.deb
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub

# Install Python and some utilities
RUN apt-get update && apt install -y --no-install-recommends \
    git \
    build-essential \
    python3-dev \
    python3-pip \
    python3-setuptools \
    sudo 2>&1

#RUN useradd -m ubuntu 
#RUN useradd -m /home/tkonuk -s /bin/bash -g root -G sudo -u 1001 tkonuk
RUN adduser --disabled-password --gecos '' ubuntu
RUN adduser ubuntu sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

RUN mkdir /tmp/mm
#RUN chown -R ubuntu:ubuntu /tmp/mm
#RUN chown -R ubuntu:ubuntu /home/ubuntu/

#COPY --chown=ubuntu . /home/ubuntu/app

#USER ubuntu

# Install all the basic packages 
RUN pip3 install \
    # Numpy and Pandas 
    numpy pandas matplotlib \
    # PyTorch and torchvision
    #torch torchvision torchtext --extra-index-url https://download.pytorch.org/whl/cu113 \
    torchvision torchtext \
    # FairScale for fully sharded parallel training
    fairscale \
    # HuggingFace datasets and transformers
    datasets transformers 2>&1


WORKDIR /workspace

COPY . /workspace

#RUN pip3 install -r /home/ubuntu/app/requirements.txt & pip3 install -r /home/ubuntu/app/examples/flava/requirements.txt 2>&1
#RUN pip3 install -r requirements.txt & pip3 install -r examples/flava/requirements.txt 2>&1

#RUN sudo python3 setup.py install
RUN ./install_req.sh


#WORKDIR /home/ubuntu/app/examples/flava

#ENTRYPOINT ["/bin/bash"]

# build with
#   docker build --network=host -t multimodal:training .
# run with 
#   docker run --gpus all \
#                -v /datasets:/datasets \
#               -v /tmp/mm:/tmp/mm \
#               -e TRANSFORMERS_CACHE=/tmp/multimodal \
#              --ipc=host --network=host -ti multimodal:training 
# enjoy with
#   a glass of single-malt whiskey and Belgian chocolate

