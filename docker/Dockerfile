FROM ubuntu:20.04


####################################################################################
###  Basic packages
####################################################################################

RUN apt-get update 
RUN apt-get update --fix-missing
RUN apt-get upgrade -y
RUN DEBIAN_FRONTEND="noninteractive" apt-get install -y make autoconf automake libtool shtool wget curl \
                       xz-utils gcc g++ cmake \
                       ninja-build make python \
                       build-essential git ca-certificates \
                       tar gzip vim \
                       python3-pip python3-venv \
                       graphviz graphviz-dev \
                       software-properties-common


####################################################################################
## Install racket and rosette
####################################################################################

RUN add-apt-repository -y ppa:plt/racket && \
    apt-get update && \
    apt-get install -y racket racket-doc

RUN raco pkg install --deps "search-auto" rosette || echo "Ignore Error"

####################################################################################
### Copy and unzip repository
####################################################################################

RUN mkdir -p /synthesis/synthCT
COPY synthCT.tar.gz /synthesis/
RUN cd /synthesis && \
    tar -xvf synthCT.tar.gz -C /synthesis/synthCT/

###################################################################################
### Install requirements from pip
###################################################################################

RUN cd /synthesis/synthCT/ && \
    pip3 install -r requirements.txt && \
    pip3 install loguru

##################################################################################
### Setup more directories and copy config files
##################################################################################

COPY X86-64-semantics.tar.gz /synthesis
RUN tar -xf /synthesis/X86-64-semantics.tar.gz -C /synthesis/synthCT

RUN mkdir -p /synthesis/synthCT/synth-results
RUN mkdir -p /synthesis/synthCT/rosette/inst_sems
COPY pipe_config.yaml /synthesis/synthCT/

ARG conn="127.0.0.1"
ENV LOGURU_LEVEL='SUCCESS'
RUN -i "s/127.0.0.1/$conn/g" /synthesis/synthCT/conn.yaml

WORKDIR /synthesis/synthCT
CMD python3 -m synthesis.synthesis \
    --isa ./third-party/X86-64-semantics/semantics/registerInstructions/*.k \
    --pseudo-inst ./synthesis/pseudo.yaml \
    --selector "knn" \
    --parallel-tasks 75 \
    --timeout 1200 \
    --try-instruction-factorization \
    --factorizer='bottom_up' \
    --conn conn.yaml
