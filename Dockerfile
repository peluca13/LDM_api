FROM python:3.8.13-slim

COPY * ./

RUN apt-get update
RUN apt install -y vim
RUN pip install -r dependencies.txt

ENTRYPOINT ["tail", "-f", "/dev/null"]