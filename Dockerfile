FROM piotrekzie100/horovod:0.20.0-pytorch1.6.0-py38-cuda10.2
ARG PYPI_USERNAME
ARG PYPI_PASSWORD
ADD dist/* ./
RUN pip install *.whl --extra-index-url=https://${PYPI_USERNAME}:${PYPI_PASSWORD}@pypi.trasee.io/simple
RUN rm -rf *.whl
COPY assets/pretrained/ /app/assets/pretrained/
WORKDIR /app
