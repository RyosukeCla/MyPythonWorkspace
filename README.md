# My Python Workspace
My research workspace for graduation thesis

## Feature
|Jupyter Kernels|
|:-------------:|
| python 3.6.x  |
| R 3.4.x       |
| Julia 0.6.2   |

includings
- pyenv
- anaconda 3
- jupyterlab
- jupyter nbextensions
- nodejs
- julia
- r

pacakges
- pymc3
- tensorflow / keras / chainer / theano / scikit learn
- scikit image / opencv
- numpy / scipy
- pandas
- matplotlib / seaborn
- etc

## Requirement
Install Docker Community Edittion.
- https://www.docker.com/community-edition

## Getting Started

### launch workspace
```
$ docker-compose up -d
```

### boot jupyterlab
```
$ docker-compose exec admin jupyter lab
```
go to [localhost:8080](http://localhost:8080)

### boot tensor board
```
$ docker-compose exec admin tensorboard --logdir=./path/to/logs
```
go to [localhost:6006](http://localhost:6006)

### destory workspace
```
$ docker-compose down
```

### start/stop worksplace
if you want to stop workspace temporarily,
```
$ docker-compose stop
```

and, if you want to start stopped workspace
```
$ docker-compose start
```
