#! /bin/bash

echo creating a virtual env for posting request through python script "./post_request.py"
python3 -m venv .venv

echo activating venv
source .venv/bin/activate

echo installing required packages
pip install requests      

echo example usage: 'python3 post_request.py ./path/to/image/file.jpg'