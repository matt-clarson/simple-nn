# MNIST Classifier
Use the `simplenn` package to train a classifier for the MNIST hand-written digits.

## Use
To use, make sure you have everything installed (`pip install requirements.txt`).  
Then, download the MNIST dataset, by running the `get_mnist_data.sh` script. THis will create a `data` directory and download the dataset into it, as well as preparing it for use in Python (i.e. creating NumPy arrays for the dataset, and saving them to `pickle` files). You will be prompted to provide a domain for the dataset, the default is http://yann.lecun.com/exdb/mnist/ - but you can provide your own if you have the datasets hosted on a mirror somewhere.

Modify the params to use in the `simplenn-params.json` file, and then train the network by running `run.py` (for information on the params and what they do, see the `simplenn` readme).
