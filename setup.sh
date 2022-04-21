#!/bin/bash

git submodule init
git submodule update

if [ -d "./.synth-venv" ]
then
    echo "virtualenv already exists! Re-using."
else
    python3 -m venv .synth-venv
fi

source .synth-venv/bin/activate

mkdir -p ./rosette/inst_sems/

pip install wheel
pip install --upgrade pip
pip install -r requirements.txt

echo "virtualenv setup and ready to go!"
echo 'Please activate virtualenv: `source .synth-venv/bin/activate` before starting'
