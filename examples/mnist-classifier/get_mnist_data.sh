#!/bin/bash

set -o errexit;

## exit codes
OK=0;
BAD_FILE=1;

## const variables
# file names
TRAIN_IMAGES="train-images-idx3-ubyte.gz";
TRAIN_LABELS="train-labels-idx1-ubyte.gz";
TEST_IMAGES="t10k-images-idx3-ubyte.gz";
TEST_LABELS="t10k-labels-idx1-ubyte.gz";
# md5sums
TRAIN_IMAGES_MD5="f68b3c2dcbeaaa9fbdd348bbdeb94873";
TRAIN_LABELS_MD5="d53e105ee54ea40749a09fcbcd1e9432";
TEST_IMAGES_MD5="9fb629c4189551a2d022fa330f9573f3";
TEST_LABELS_MD5="ec29112dd5afa0611ce80d1b7f02629c";


if [ -z !$1 ] && [ $1 == "--help" ]; then
  echo "help goes here";
  exit $OK;
fi

MNIST_DOMAIN=${1:-"http://yann.lecun.com/exdb/mnist/"};

echo "Using domain '$MNIST_DOMAIN' to get MNIST data";
read -n 1 -p "Is this ok? (y/[n]): " proceed;

if [ -z $proceed ] || [ $proceed != "y" ]; then
  echo -e "\nQuitting\n";
  exit $OK;
fi

echo "";


if [ -d "data" ]; then
  echo "Removing old './data' directory";
  rm -rf ./data;
fi

echo "Creating './data' directory";
mkdir data;

echo "Changing into new working directory './data'";
pushd data;

echo "Starting downloads";
wget -q --show-progress "$MNIST_DOMAIN/$TRAIN_IMAGES";
wget -q --show-progress "$MNIST_DOMAIN/$TRAIN_LABELS";
wget -q --show-progress "$MNIST_DOMAIN/$TEST_IMAGES";
wget -q --show-progress "$MNIST_DOMAIN/$TEST_LABELS";

get_sum() {
  echo $(md5sum $1 | awk '{ print $1 }');
}

echo "Verifying files";
if [ $(get_sum $TRAIN_IMAGES) != $TRAIN_IMAGES_MD5 ] || [ $(get_sum $TRAIN_LABELS) != $TRAIN_LABELS_MD5 ] || [ $(get_sum $TEST_IMAGES) != $TEST_IMAGES_MD5 ] || [ $(get_sum $TEST_LABELS) != $TEST_LABELS_MD5 ]; then
  echo "Bad file found";
  exit BAD_FILE;
fi

echo "Unzipping MNIST data files";
gunzip $TRAIN_IMAGES;
gunzip $TRAIN_LABELS;
gunzip $TEST_IMAGES;
gunzip $TEST_LABELS;

echo "Preparing files for use with python";
python ../parse_data.py *;

echo "Restoring working directory";
popd;
echo "Done";
exit $OK;


