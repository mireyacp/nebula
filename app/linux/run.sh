#!/bin/bash

# Change to the directory one level up
cd ../..

# Activate the virtual environment
source .venv/bin/activate

cd app/

# Run the Python script
python main.py
