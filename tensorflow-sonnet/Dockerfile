FROM tensorflow/tensorflow:latest-gpu-py3

COPY ./code/ /usr/src/app
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /usr/src/app/

RUN DEBIAN_FRONTEND=noninteractive apt install -y tzdata

#installieren von updates
RUN apt-get update && apt-get install -y
RUN apt-get install wget git-all cmake vim apt python2.7 python-pip -y

RUN pip2 install numpy
#required installs for Opencv
RUN apt-get install build-essential cmake unzip pkg-config -y && \
    apt-get install libjpeg-dev libpng-dev libtiff-dev -y && \
    apt-get install libavcodec-dev libavformat-dev libswscale-dev -y && \
    apt-get install libxvidcore-dev libx264-dev -y && \
    apt-get install libgtk-3-dev -y && \
    apt-get install libatlas-base-dev gfortran -y && \
    apt-get install python3-dev -y && \
    apt-get install libavresample-dev -y && \
    apt-get install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev -y && \
    apt-get install libv4l-dev -y && \
    apt-get install v4l-utils -y && \
    apt-get install libdc1394-22-dev -y

#zusätzliche installs vlt nicht nötig
RUN apt-get install libblkid-dev e2fslibs-dev libboost-all-dev libaudit-dev -y && \
    apt-get install build-essential libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev -y
RUN apt-get update && apt-get install -y

#RUN make install 

#installieren von requirements for python
WORKDIR /usr/src/app/
RUN pip install --trusted-host pypi.python.org -r requirements.txt