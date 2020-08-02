FROM anibali/pytorch:1.5.0-cuda10.2
USER root
ARG PYPI_USERNAME=trasee_rd
ARG PYPI_PASSWORD
ARG SSDIR_VERSION=0.1.0
ARG UID=1000
ARG GID=1000
RUN apt-get update -yqq && apt-get install -yqq libglib2.0-0
RUN pip install -i https://${PYPI_USERNAME}:${PYPI_PASSWORD}@pypi.trasee.io/simple/ ssdir==${SSDIR_VERSION}
RUN chown -R ${UID}:${GID} /app
USER ${UID}:${GID}
WORKDIR /app
ENTRYPOINT ["ssdir"]
