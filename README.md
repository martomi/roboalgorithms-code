# roboalgorithms-code
Collection of code examples to go along with articles on roboalgorithms.com

# Python Notebooks

All python notebooks can be found under the `notebooks` folder.

## How to run?

### With Docker (Recommended)

The easiest way is to run inside Docker [jupyter/scipy-notebook](https://hub.docker.com/r/jupyter/scipy-notebook) container.
It contains all necessary dependencies. You'll only need to mount the `notebooks` folder from this repository to the
`/home/jovyan/work` inside the container.

Here's an example command sequence:
```
git clone https://github.com/martomi/roboalgorithms-code.git
cd roboalgorithms-code/notebooks
docker run --rm -p 8888:8888 -v "$PWD":/home/jovyan/work jupyter/scipy-notebook:399cbb986c6b
```

After that you'll see a link similar to that one in the console that you can open in the browser to interact with the
notebook:
```
http://127.0.0.1:8888/?token=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### Without Docker

```
TBD
```

## External Datasets

If the notebook requires additional data, you'll need to follow the links from the article and download it into
the `notebooks/data` folder.
