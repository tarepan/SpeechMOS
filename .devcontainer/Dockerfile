# derived from Microsoft Python dev container (https://github.com/microsoft/vscode-dev-containers/tree/main/containers/python-3)
FROM mcr.microsoft.com/vscode/devcontainers/python:3.10

# Colab versions @2023-05-06
# Python:                 3.10.11
# NumPy:      numpy       1.22.4
# PyTorch:    torch       2.0.0+cu118
# Torchaudio: torchaudio  2.0.1+cu118

# PyTorch on CPU
RUN pip3 --disable-pip-version-check --no-cache-dir install torch==2.0.0+cpu torchaudio==2.0.1+cpu -f https://download.pytorch.org/whl/torch_stable.html

# package dependency manager `poetry` without environment isolation (It is a job of Docker.)
ENV POETRY_VIRTUALENVS_CREATE=false
RUN curl -sSL https://install.python-poetry.org | python3 -