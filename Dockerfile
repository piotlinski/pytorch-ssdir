FROM piotrekzie100/horovod:0.20.0-pytorch1.6.0-py38-cuda10.2
ADD dist/* ./
RUN pip install *.whl
RUN rm -rf *.whl
COPY assets/pretrained/vgglite_mnist_sc_SSD-VGGLite_MultiscaleMNIST/ /app/assets/pretrained/vgglite_mnist_sc_SSD-VGGLite_MultiscaleMNIST/
WORKDIR /app
