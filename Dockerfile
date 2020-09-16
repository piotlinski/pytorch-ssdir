FROM piotrekzie100/pytorch:1.6.0-py38-cuda10.2-horovod0.20.0
ADD dist/* ./
RUN pip install *.whl
RUN rm -rf *.whl
COPY assets/pretrained/vgglite_mnist_sc_SSD-VGGLite_MultiscaleMNIST/ /app/assets/pretrained/vgglite_mnist_sc_SSD-VGGLite_MultiscaleMNIST/
WORKDIR /app
ENTRYPOINT ["horovodrun"]
