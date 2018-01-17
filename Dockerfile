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
ENV CONDA_DIR $PYENV_ROOT/versions/anaconda3-2.5.0/
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
RUN conda install -y -c conda-forge jupyterlab

# install nodejs
RUN conda install -y nodejs

# install R
USER root
RUN apt-get update && \
  apt-get install -y --no-install-recommends \
  fonts-dejavu \
  tzdata \
  gfortran \
  gcc \
  libxext-dev \
  libxrender1 && apt-get clean && \
  rm -rf /var/lib/apt/lists/*

# install R kernel for jupyter
USER admin
RUN conda config --system --append channels r && \
  conda install --quiet --yes \
  'rpy2' \
  'r-base' \
  'r-irkernel' \
  'r-plyr' \
  'r-devtools' \
  'r-tidyverse' \
  'r-shiny' \
  'r-rmarkdown' \
  'r-forecast' \
  'r-rsqlite' \
  'r-reshape2' \
  'r-nycflights13' \
  'r-caret' \
  'r-rcurl' \
  'r-crayon' \
  'r-randomforest' && \
  conda clean -tipsy

# install Julia
USER root
ENV JULIA_PKGDIR=/opt/julia
ENV JULIA_VERSION=0.6.2
RUN mkdir /opt/julia-${JULIA_VERSION} && \
    cd /tmp && \
    wget -q https://julialang-s3.julialang.org/bin/linux/x64/`echo ${JULIA_VERSION} | cut -d. -f 1,2`/julia-${JULIA_VERSION}-linux-x86_64.tar.gz && \
    echo "dc6ec0b13551ce78083a5849268b20684421d46a7ec46b17ec1fab88a5078580 *julia-${JULIA_VERSION}-linux-x86_64.tar.gz" | sha256sum -c - && \
    tar xzf julia-${JULIA_VERSION}-linux-x86_64.tar.gz -C /opt/julia-${JULIA_VERSION} --strip-components=1 && \
    rm /tmp/julia-${JULIA_VERSION}-linux-x86_64.tar.gz
RUN ln -fs /opt/julia-*/bin/julia /usr/local/bin/julia

RUN mkdir /etc/julia && \
    echo "push!(Libdl.DL_LOAD_PATH, \"$CONDA_DIR/lib\")" >> /etc/julia/juliarc.jl && \
    mkdir $JULIA_PKGDIR && \
    chown admin $JULIA_PKGDIR

# install Julia kernel for jupyter
USER admin
RUN \
  julia -e 'Pkg.init()' && \
  julia -e 'Pkg.update()' && \
  julia -e 'Pkg.add("IJulia")' && \
  julia -e 'using IJulia'

# install DNN lib
RUN conda install -y tensorflow keras chainer
RUN conda install -y opencv

# install mcmc lib
RUN conda install -y pymc3 mkl theano pygpu
ENV MKL_THREADING_LAYER=GNU
