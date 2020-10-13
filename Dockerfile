FROM piotrekzie100/horovod:0.20.0-pytorch1.6.0-py38-cuda10.2
ADD dist/* ./
RUN pip install *.whl
RUN rm -rf *.whl
COPY assets/pretrained/ /app/assets/pretrained/
WORKDIR /app
