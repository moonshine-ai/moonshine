# Moonshine

## Setup

* Install `uv` for Python environment management
  
  - Follow instructions [here](https://github.com/astral-sh/uv)

* Create and activate virtual environment
  
  ```shell
    uv venv env_moonshine
    source env_moonshine/bin/activate
  ```

* Install the `useful-moonshine` package from this github repo
  
  ```shell
  uv pip install useful-moonshine@git+https://github.com/usefulsensors/moonshine.git
  ```
  
  `moonshine` inference code is written in Keras and can run with the backends that Keras supports. The about command install with the default `tensorflow` backend. To install and run with PyTorch backend, run the following :
  
  ```shell
  uv pip install useful-moonshine[torch]@git+https://github.com/usefulsensors/moonshine.git
  export KERAS_BACKEND=torch
  ```

  To run with jax backend, run the following:

  ```shell
  uv pip install useful-moonshine[jax]@git+https://github.com/usefulsensors/moonshine.git
  export KERAS_BACKEND=jax
  # Use useful-moonshine[jax-cuda] for jax on GPU
  ```

  * _Note for UsefulSensors: Since the repo is not public yet, installing from the github URI is not possible yet. Do this instead:_

  * ```shell
    git clone git@github.com:usefulsensors/moonshine.git
    cd moonshine
    uv pip install -e .
    cd ../
    # Make sure you are out of the current directory too, as you
    # don't want moonshine in your path (import moonshine in python
    # will mistakenly use the wrong path)
    ```

* Test transcribing an audio file
  
  ```shell
  python
  >>> import moonshine
  >>> moonshine.transcribe(moonshine.ASSETS_DIR / 'beckett.wav', 'moonshine/tiny')
  ['Ever tried ever failed, no matter try again, fail again, fail better.']
  ```

  * The first argument is the filename for an audio file, the second is the name of a moonshine model. `moonshine/tiny` and `moonshine/base` are the currently available models.
