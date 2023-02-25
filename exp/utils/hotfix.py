"""For some reason, the MNIST dataset is not available via
train_data = MNIST(root = './', train=True, download=True, transform=transform).
One solution is to download the data manually first.
"""

import os
import tarfile
import urllib.request


def download_MNIST_hotfix(path):
    """
    Manually download MNIST data if necessary
    """

    DATA_FOLDER_NAME = "data"
    MNIST_FOLDER_NAME = "MNIST"
    MNIST_SUBFOLDER1_NAME = "raw"
    MNIST_SUBFOLDER2_NAME = "processed"
    URL = "http://www.di.ens.fr/~lelarge/MNIST.tar.gz"
    FILE_NAME = "MNIST.tar.gz"

    # Create paths
    DATA_PATH = os.path.join(path, DATA_FOLDER_NAME)
    MNIST_PATH = os.path.join(DATA_PATH, MNIST_FOLDER_NAME)
    MNIST_SUBFOLDER1_PATH = os.path.join(MNIST_PATH, MNIST_SUBFOLDER1_NAME)
    MNIST_SUBFOLDER2_PATH = os.path.join(MNIST_PATH, MNIST_SUBFOLDER2_NAME)

    # Recursively create all folders needed to create MNIST subfolders
    os.makedirs(MNIST_SUBFOLDER1_PATH, exist_ok=True)
    os.makedirs(MNIST_SUBFOLDER2_PATH, exist_ok=True)

    # Check if MNIST subfolders actually contains data
    if (
        len(os.listdir(MNIST_SUBFOLDER1_PATH)) == 0
        or len(os.listdir(MNIST_SUBFOLDER2_PATH)) == 0
    ):
        # Download data
        PATH_TO_FILE = os.path.join(DATA_PATH, FILE_NAME)
        urllib.request.urlretrieve(URL, PATH_TO_FILE)

        # Extract files
        tar = tarfile.open(PATH_TO_FILE, "r:gz")
        tar.extractall(path=DATA_PATH)
        tar.close()

        # Delete MNIST.tar.gz
        os.remove(PATH_TO_FILE)
