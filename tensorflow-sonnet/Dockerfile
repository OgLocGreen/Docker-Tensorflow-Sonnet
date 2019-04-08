FROM tensorflow/tensorflow:latest-gpu

COPY ./code/ /usr/src/app

WORKDIR /usr/src/app/

#installieren von updates
RUN apt-get update && apt-get install -y
RUN apt-get install wget git-all cmake vim -y

#install opencv from source
RUN mkdir opencv
WORKDIR /usr/src/app/opencv/
RUN git clone https://github.com/opencv/opencv.git
RUN git clone https://github.com/opencv/opencv_contrib.git
#RUN cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local ..
#RUN make -j4
#RUN make install

#installieren von Denseflow
WORKDIR /usr/src/app/
RUN mkdir denseflow
WORKDIR /usr/src/app/denseflow/
RUN git clone --recursive http://github.com/yjxiong/dense_flow
RUN mkdir build
WORKDIR /usr/src/app/denseflow/
#RUN cmake .. && make -j

#installieren von requirements for python
WORKDIR /usr/src/app/
RUN pip install --trusted-host pypi.python.org -r requirements.txt

#RUN git clone https://essgitlab.fzi.de/szhang/ma_shuozhang/tree/master/live_test
