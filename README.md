# Docker-compose

Dieses Projekt beschäftigt sich mit docker und docker-compose.

## Getting Started
Es werden Beispieldateien für ein Docker-compose gegeben mit wechelen man sich einfach einen eigenen Docker Container zusammenbauen kann:
* Nvidia-smi
* Tensorflow-gpu
* Sonnet


### Prerequisites
Man muss als erstes Docker und Docker-Compose installieren.
Wenn Tensorflow mit GPU unterstützung verweden werden soll, müssen auch die entsprechenden Treiber für die Grafikkarte installiert werden.

### Installing
Anschließend kann man sich das Reposetrie clonen.

Und mit folgendem Befehl sich sein Docker-image bauen:
```
Sudo docker-compose build
```
und mit: 
```
Sudo docker-compose up
```
Sein Container starten.


