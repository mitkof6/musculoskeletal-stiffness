FROM ubuntu:18.04

# required for tzdata
ENV DEBIAN_FRONTEND=noninteractive

# install ubuntu packages
RUN apt-get update && apt-get -y upgrade && apt-get -y \
install imagemagick python3 python3-pip cython libgmp-dev \
lrslib git python3-tk liblapack-dev freeglut3-dev libxi-dev \
libxmu-dev doxygen swig python-dev g++ openjdk-11-jdk \
cmake-curses-gui tzdata nano wget
RUN alias python=python3

# setup timezone
RUN ln -fs /usr/share/zoneinfo/Europe/Athens /etc/localtime
RUN dpkg-reconfigure --frontend noninteractive tzdata

# setup python dependencies
RUN pip3 install scipy numpy sympy pandas seaborn matplotlib \
tqdm multipolyfit regex jupyter pycddlib
RUN pip3 install pydy  # problem with pydy for python3

# install and setup OpenSim
RUN wget https://sourceforge.net/projects/dependencies/files/opensim-core/opensim-core-ubunut18.04.tar.xz
RUN tar -xvf opensim-core-ubunut18.04.tar.xz /root/
ENV OPENSIM_HOME=/root/opensim-core
ENV OpenSim_DIR=$OPENSIM_HOME/lib/cmake/OpenSim
ENV LD_LIBRARY_PATH=$OPENSIM_HOME/lib:$LD_LIBRARY_PATH
ENV PATH=$OPENSIM_HOME/libexec/simbody:$PATH
ENV PATH=$OPENSIM_HOME/bin:$PATH
RUN cd /root/opensim-core/lib/python3.6/site-packages \
&& python setup.py install

# copy demos to default directory
ADD arm_model.tar.xz /root/
ADD feasible_joint_stiffness.tar.xz /root/
ADD data.tar.xz /root/

# make port 80 available to the outside
EXPOSE 80

# setup jupyter (with password 'stanev')
RUN cd && mkdir .jupyter
ADD jupyter_notebook_config.json /root/.jupyter/jupyter_notebook_config.json
CMD cd && jupyter notebook --ip=0.0.0.0 --port=80 --allow-root

