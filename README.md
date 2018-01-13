# My Python Workspace
Build python environment without any annoying protocols.
All you need is Docker.

including
- pyenv
- anaconda 3
- jupyterlab
- jupyter with extensions
- nodejs

## setup workspace
```
$ docker-compose up -d
```

## jupyterlab
```
$ docker-compose exec admin bash jupyter lab
```
go to [localhost:8080](http://localhost:8080)

## destory workspace
```
$ docker-compose down
```

## start/stop worksplace
```
$ docker-compose start
$ docker-compose stop
```
