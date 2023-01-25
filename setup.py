
from setuptools import setup, find_packages
from graph_embs_algos import __version__ as version

setup_params = {
    "name": "graph_embs_algos",
    "version": version,
    "description": "A python library for creating graph embeddings via PyTorch.",
    "url": "https://github.com/ron1x1-abba/GraphEmbeddingsAlgos",
    "author": "Golikov Artemii",
    "license": "Apache 2.0",
    "packages": find_packages(),
    "include_package_data": True,
    "zip_safe": False,
    "install_requires": []
}

if __name__ == "__main__":
    setup(**setup_params)
