FROM ubuntu:16.04

# install essentials
USER root
RUN \
  sed -i 's/# \(.*multiverse$\)/\1/g' /etc/apt/sources.list && \
  apt-get update && \
  apt-get -y upgrade && \
  apt-get install -y build-essential && \
  apt-get install -y software-properties-common && \
  apt-get install -y byobu curl git htop man unzip vim wget && \
  rm -rf /var/lib/apt/lists/*

# regist user
RUN useradd -ms /bin/bash admin
USER admin
WORKDIR /home/admin/workspace

# install pyenv
RUN git clone https://github.com/yyuu/pyenv.git ~/.pyenv
ENV HOME  /home/admin
ENV PYENV_ROOT $HOME/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH

# install anaconda3
RUN \
  pyenv install anaconda3-5.0.1 && \
  pyenv rehash && \
  pyenv global anaconda3-5.0.1
ENV PATH $PYENV_ROOT/versions/anaconda3-2.5.0/bin/:$PATH
RUN conda update conda

# setup jupyter
RUN \
  jupyter-notebook --generate-config && \
  echo "c = get_config()" >> ~/.jupyter/jupyter_notebook_config.py && \
  echo "c.NotebookApp.ip = '0.0.0.0'" >> ~/.jupyter/jupyter_notebook_config.py && \
  echo "c.NotebookApp.open_browser = False" >> ~/.jupyter/jupyter_notebook_config.py && \
  echo "c.NotebookApp.port = 8080" >> ~/.jupyter/jupyter_notebook_config.py && \
  echo "c.NotebookApp.token = ''" >> ~/.jupyter/jupyter_notebook_config.py

# install jupyter extensions
RUN \
  pip install jupyter_contrib_nbextensions && \
  jupyter contrib nbextension install --user && \
  jupyter nbextension enable codefolding/main && \
  jupyter contrib nbextensions migrate

# install jupyter lab
RUN conda install -y conda-forge jupyterlab

# install nodejs
RUN conda install -y conda-forge nodejs