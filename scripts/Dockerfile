# The base image fixes the glibc that everything built here links against, which
# is the floor a wheel's manylinux tag records and the oldest system it will
# install on. bookworm (2.36) covers current distributions; build-pip-docker.sh
# also builds this image on bullseye (2.31) for the arm64 wheel, so Raspberry Pi
# OS bullseye and its contemporaries can still install us.
ARG BASE_IMAGE=python:3.12-slim-bookworm
FROM ${BASE_IMAGE}

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Installl essential build tools
RUN apt-get update && \
    apt-get install -y \
    build-essential \
    curl

# cmake and patchelf come from PyPI rather than apt because bullseye's are too
# old for us -- cmake 3.18 against the 3.22 core/CMakeLists.txt asks for, and
# patchelf 0.12 against the 0.14 auditwheel requires to repair a wheel -- and
# this way every base image builds with the same ones. cmake is held below 4,
# which refuses to configure projects that declare a minimum older than 3.5,
# as some of the vendored ones do.
RUN pip install --no-cache-dir "cmake<4" patchelf

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Create a directory in home for the repo
RUN mkdir -p /home/user/moonshine

# Set the working directory
WORKDIR /home/user/moonshine

# Create a non-root user (optional but recommended)
RUN useradd -m -s /bin/bash user && \
    chown -R user:user /home/user

# Switch to non-root user
USER user

# Set Python 3 as default python
RUN echo 'alias python=python3' >> ~/.bashrc && \
    echo 'alias pip=pip3' >> ~/.bashrc

# Copy .pypirc configuration file from build context to Docker image user's home directory
# Note: The .pypirc file must be in the build context (project root) for this to work.
# To use your home directory .pypirc, copy it first: cp ~/.pypirc .pypirc
# Then build from the repo root: docker build -f scripts/Dockerfile -t <image> .
COPY --chown=user:user .pypirc /home/user/.pypirc

# Default command
CMD ["/bin/bash"]

