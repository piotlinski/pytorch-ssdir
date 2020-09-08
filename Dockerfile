FROM anibali/pytorch:1.5.0-cuda10.2
USER root
ARG UID=1000
ARG GID=1000
RUN apt-get update -yqq && apt-get install -yqq libglib2.0-0 gcc
ADD dist/* ./
RUN pip install *.whl
RUN rm -rf *.whl
COPY assets/pretrained/vgglite_mnist_sc_SSD-VGGLite_MultiscaleMNIST/ /app/assets/pretrained/vgglite_mnist_sc_SSD-VGGLite_MultiscaleMNIST/
RUN chown -R ${UID}:${GID} assets/
USER ${UID}:${GID}
WORKDIR /app
ENTRYPOINT ["ssdir"]
