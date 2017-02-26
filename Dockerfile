FROM gw000/keras:1.2.1-py3-tf-cpu

RUN apt-get update && apt-get install --no-install-recommends -y python3-pip python-pip \
 python3-dev python3-pydot python-pillow python3-pillow python-dev \
&& apt-get install -y python-setuptools python3-setuptools

RUN pip --no-cache-dir install \
    # jupyter notebook and ipython (Python 2)
    ipython \
    ipykernel \
    jupyter \
&& python -m ipykernel.kernelspec

RUN pip3 --no-cache-dir install \
    # jupyter notebook and ipython (Python 3)
    ipython \
    ipykernel \
 && python3 -m ipykernel.kernelspec

RUN apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# for jupyter
EXPOSE 8888
# for tensorboard
EXPOSE 6006

WORKDIR /srv/
