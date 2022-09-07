FROM python:3.8.13-slim

COPY * ./

RUN apt-get update
RUN pip install -r dependencies.txt

ENTRYPOINT ["tail", "-f", "/dev/null"]