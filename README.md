# Docker-compose

Dieses Projekt beschäftigt sich mit docker und docker-compose.

## Getting Started
Es werden Beispieldateien für ein Docker-compose gegeben mit wechelen man sich einfach einen eigenen Docker Container zusammenbauen kann:
* Nvidia-smi
* Tensorflow-gpu
* Sonnet


### Prerequisites
Um die genangen Images/Container bauen und ausführen zukönnen müssen Folgende Programme installiert werden:
* [Docker2.0](https://docs.docker.com/install/linux/docker-ce/ubuntu/)
* [Nvida-Docker](https://github.com/NVIDIA/nvidia-docker) - Nvidia-Docker nur für nvidia-runtime anwendungen
* [Docker-Compose](https://docs.docker.com/compose/install/)

(Wenn Tensorflow mit GPU unterstützung verweden werden soll, müssen auch die entsprechenden Treiber für die Grafikkarte installiert werden.)

### Installing
Anschließend kann man sich das Reposetrie clonen.

### Using
Und mit folgendem Befehl sich sein Docker-image bauen:

```
Sudo docker-compose build
```

und mit: 

```
Sudo docker-compose up
```

Sein Container starten. Dann luft das im docker-compose ausgewählte Command solange bis es fertig ausgeführt ist.
Wenn das POrgramm dauerhaft laufen soll muss sich es in einer While(true) Schleifen-rotine laufen.


