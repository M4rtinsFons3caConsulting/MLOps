ARG BASE_IMAGE=python:3.11-slim
FROM $BASE_IMAGE as runtime-environment

# update pip and install uv
RUN python -m pip install -U "pip>=21.2"
RUN pip install uv

# install project requirements
COPY requirements.txt /tmp/requirements.txt
RUN apt-get update && apt-get install -y build-essential gcc
RUN uv pip install --system --no-cache-dir -r /tmp/requirements.txt && rm -f /tmp/requirements.txt

# add kedro user
ARG KEDRO_UID=999
ARG KEDRO_GID=0
RUN groupadd -f -g ${KEDRO_GID} kedro_group && \
    useradd -m -d /home/kedro_docker -s /bin/bash -g ${KEDRO_GID} -u ${KEDRO_UID} kedro_docker

WORKDIR /home/kedro_docker
USER kedro_docker

FROM runtime-environment

# copy the whole project except what is in .dockerignore
ARG KEDRO_UID=999
ARG KEDRO_GID=0
COPY --chown=${KEDRO_UID}:${KEDRO_GID} . .
COPY --chown=${KEDRO_UID}:${KEDRO_GID} conf/local/credentials.yml /home/kedro_docker/conf/local/credentials.yml

# Fix permissions
RUN find /home/kedro_docker -type d -exec chmod 755 {} \; && \
    find /home/kedro_docker -type f -exec chmod 644 {} \;

EXPOSE 8888

CMD ["kedro", "run"]
