FROM python:3.12-slim-bookworm

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Installl essential build tools
RUN apt-get update && \
    apt-get install -y \
    cmake \
    build-essential \
    patchelf

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
# Then build: docker build -t <image> .
COPY --chown=user:user .pypirc /home/user/.pypirc

# Default command
CMD ["/bin/bash"]

