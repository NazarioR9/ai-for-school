FROM ubuntu:latest
LABEL Name = UBUNTU LASTEST

ARG DEBIAN_FRONTEND=noninteractive
ENV TERM=xterm
ENV TZ=Europe/Paris

RUN apt-get update \
    && apt-get install -yqq \
        git git-core \
        gcc make autoconf libc-dev pkg-config \
        curl \
        wget \
        cron \
        strace \
        build-essential \
        cmake \
        python3-pandas \
        python3-pip \
        python3-numpy \
        unzip \
        sudo \
        nano \
        ssh \
        && apt-get clean \
        && echo 'alias python="python3"' >> ~/.bashrc \
	&& echo 'alias pip="pip3"' >> ~/.bashrc

COPY requirements.txt .
RUN pip3 install -U -r requirements.txt

RUN groupadd appuser \
	&& useradd -m -r -u 1000 -g appuser appuser \
	&& usermod -aG sudo appuser \
	&& mkdir -p /home/appuser/ \
	&& chown -R appuser:appuser /home/appuser/ \
	&& chmod -R 0711 /home/appuser/ \
	&& ln -s /usr/bin/python3.8 /usr/bin/python \
	&& echo 'export PATH=$HOME/bin/:usr/local/bin:/usr/local/sbin:/home/appuser/.local/bin:$PATH' >> /home/appuser/.bashrc \
	&& echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER appuser

WORKDIR /home/appuser

RUN mkdir .streamlit

EXPOSE 80
EXPOSE 8501
