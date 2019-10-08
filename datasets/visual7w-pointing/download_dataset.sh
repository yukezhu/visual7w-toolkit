#!/usr/bin/env bash

V7W_DB_NAME=v7w_pointing

V7W_URL="http://ai.stanford.edu/~yukez/papers/resources/dataset_${V7W_DB_NAME}.zip"
V7W_PATH="dataset_${V7W_DB_NAME}.json"

if [ -f "dataset.json" ]; then
  echo "Dataset already exists."
  read -p "Would you like to update the dataset file? [y/n] " -n 1 -r
  echo    # (optional) move to a new line
  if [[ ! $REPLY =~ ^[Yy]$ ]]
  then
    echo "Bye!"
    exit
  fi
fi

echo "Downloading ${V7W_DB_NAME} dataset..."
wget -q $V7W_URL -O dataset.zip
unzip -j dataset.zip
rm dataset.zip
mv $V7W_PATH dataset.json
echo "Done."
