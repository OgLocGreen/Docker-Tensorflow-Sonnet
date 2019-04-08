FROM tensorflow/tensorflow:latest-gpu-py3

COPY ./code/ /usr/src/app

WORKDIR /usr/src/app/

#installieren von updates
RUN apt-get update && apt-get install -y
RUN apt-get install wget git-all protobuf-compiler python3-pil python3-lxml python3-tk vim -y

#installieren von requirements for python
RUN pip install --trusted-host pypi.python.org -r requirements.txt

RUN git clone https://github.com/tensorflow/models.git

WORKDIR /usr/src/app/models/research/
RUN wget -O protobuf.zip https://github.com/google/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86_64.zip
RUN unzip protobuf.zip 
RUN ./bin/protoc object_detection/protos/*.proto --python_out=.
RUN export PYTHONPATH=$PYTHONPATH:/usr/src/app/models/research:/usr/src/app/models/research/slim
WORKDIR /usr/src/app/models/research/object_detection/

COPY ./webcamObjektdetection.py /usr/src/app/models/research/object_detection/webcamObjektdetection.py