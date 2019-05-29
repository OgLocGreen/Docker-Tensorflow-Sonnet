FROM tensorflow/tensorflow:latest-gpu-py3

COPY ./code/ /usr/src/app

WORKDIR /usr/src/app/

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

#install opencv from source
WORKDIR /usr/src/app
RUN mkdir opencv
WORKDIR /usr/src/app/opencv/
RUN git clone https://github.com/opencv/opencv.git --progress
WORKDIR /usr/src/app/opencv/opencv/
RUN git checkout 3.4
WORKDIR /usr/src/app/opencv/
RUN git clone https://github.com/opencv/opencv_contrib.git --progress
WORKDIR /usr/src/app/opencv/opencv_contrib/
RUN git checkout 3.4
WORKDIR /usr/src/app/opencv/opencv/
RUN mkdir build
WORKDIR /usr/src/app/opencv/opencv/build/
RUN cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/src/app/opencv/opencv/ -D INSTALL_PYTHON_EXAMPLES=OFF -D INSTALL_C_EXAMPLES=OFF -D OPENCV_EXTRA_MODULES_PATH=/usr/src/app/opencv/opencv_contrib/modules -D PYTHON3_EXECUTABLE=/usr/bin/python3.5m -D PYTHON3_LIBRARY=/usr/lib/python3.5/config-3.5m-x86_64-linux-gnu/libpython3.5m.so -D PYTHON3_INCLUDE_DIR=/usr/include/python3.5m -D PYTHON3_INCLUDE_DIR2=/usr/include/x86_64-linux-gnu/python3.4m -D PYTHON3_NUMPY_INCLUDE_DIRS=/usr/local/lib/python3.5/dist-packages/numpy/core/include/ -D PYTHON3_PACKAGES_PATH=/usr/local/lib/python3.5/dist-packages/ -D BUILD_EXAMPLES=OFF -D ENABLE_PRECOMPILED_HEADERS=OFF -D CUDA_NVCC_FLAGS=--expt-relaxed-constexpr ..
RUN make -j4
RUN make install

WORKDIR /usr/src/app/opencv/opencv/build/lib/python3/
RUN mv cv2.cpython-35m-x86_64-linux-gnu.so cv2.so

#required installs for Opencv
RUN apt-get install libzip-dev -y

#installieren von Denseflow
WORKDIR /usr/src/app/
RUN mkdir denseflow
WORKDIR /usr/src/app/denseflow/
RUN git clone --recursive http://github.com/yjxiong/dense_flow --progress
WORKDIR /usr/src/app/denseflow/dense_flow/
RUN git checkout opencv-3.1
RUN mkdir build
WORKDIR /usr/src/app/denseflow/dense_flow/build/
#RUN cmake -D OpenCV_DIR=/usr/src/app/opencv/opencv/build -D OPENCV_EXTRA_MODULES_PATH=/usr/src/app/opencv/opencv_contrib/modules ..
#RUN make -j4
#RUN make install 

#installieren von requirements for python
WORKDIR /usr/src/app/
RUN pip install --trusted-host pypi.python.org -r requirements.txt

#RUN git clone https://essgitlab.fzi.de/szhang/ma_shuozhang/tree/master/live_test
#cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/src/app/opencv/opencv/  -DOPENCV_EXTRA_MODULES_PATH=/usr/src/app/opencv/opencv_contrib/modules -D PYTHON3_EXECUTABLE=/usr/bin/python3.5m -D PYTHON3_LIBRARY=/usr/lib/python3.5/config-3.5m-x86_64-linux-gnu/libpython3.5m.so -D PYTHON3_INCLUDE_DIR=/usr/include/python3.5m -D PYTHON3_INCLUDE_DIR2=/usr/include/x86_64-linux-gnu/python3.4m -D PYTHON3_NUMPY_INCLUDE_DIRS=/usr/local/lib/python3.5/dist-packages/numpy/core/include/ -D PYTHON3_PACKAGES_PATH=/usr/local/lib/python3.5/dist-packages/ ..