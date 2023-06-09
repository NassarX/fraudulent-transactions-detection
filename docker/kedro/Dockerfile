ARG PYTHON_VERSION
ARG DEBIAN_VERSION
ARG TAG=${PYTHON_VERSION}-${DEBIAN_VERSION}

FROM python:${TAG} as base

ARG DEBIAN_FRONTEND=noninteractive

# Set the labels
LABEL maintainer="mahmoud@nassarx.com"
LABEL version="1.0"
LABEL description="Python Docker Image for Development"

# Install dependencies
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y \
        build-essential \
        cmake \
        curl \
        ca-certificates \
        gcc \
        locales \
        wget \
        git \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
ENV APP_HOME=/home/kedro_docker
WORKDIR $APP_HOME

# add kedro user
ARG KEDRO_UID=999
ARG KEDRO_GID=0
RUN groupadd -f -g ${KEDRO_GID} kedro_group && \
    useradd -m -d ${APP_HOME} -s /bin/bash -g ${KEDRO_GID} -u ${KEDRO_UID} kedro_docker

# Copy config files and install dependencies
COPY ./config/requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && \
    pip install -U -r /tmp/requirements.txt && \
    rm -f /tmp/requirements.txt

# Switch to the kedro user
USER kedro_docker

FROM base as final

# Set the working directory
ENV APP_HOME=/home/kedro_docker
WORKDIR $APP_HOME

# Fix permissions for kedro_docker user
USER root
RUN chown -R kedro_docker:${KEDRO_GID} ${APP_HOME}

# Copy kedro script
COPY ./bin/docker-entrypoint.sh /usr/local/bin/docker-entrypoint
RUN chmod +x /usr/local/bin/docker-entrypoint

# Switch to the kedro_docker user
USER kedro_docker
RUN chmod -R a+w ${APP_HOME}

# Set the environment variables
ARG PROJECT_NAME=default
ENV PROJECT_NAME="${PROJECT_NAME}"

ARG OUTPUT_DIR="."
ENV OUTPUT_DIR="${OUTPUT_DIR}"

ARG REPO_NAME=default
ENV REPO_NAME="${REPO_NAME}"

ARG PYTHON_PACKAGE=default
ENV PYTHON_PACKAGE="${PYTHON_PACKAGE}"

# Set the entrypoint
ENTRYPOINT ["docker-entrypoint"]

# Expose ports for Jupyter (8888), kedro-viz (4141)
EXPOSE 8888 4141

# Keep the container running by using an infinite loop
CMD kedro viz --host=0.0.0.0 --no-browser && tail -f /dev/null

