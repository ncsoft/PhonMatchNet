FROM tensorflow/tensorflow:2.4.1-gpu

# Install dependency
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv A4B469963BF863CC
RUN apt-get update -y && apt-get install -y \
    git \
    libsndfile1 

# Install python packages
RUN python -m pip install --upgrade pip && pip install \
    levenshtein \
    six \
    audioread \
    librosa  \
    PySoundFile \
    scipy \
    tqdm \
    pandas \
    nltk \
    inflect

RUN python -m pip uninstall -y numpy
RUN python -m pip install numpy==1.18.5

WORKDIR /home
