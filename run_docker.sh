#! /bin/bash

docker run -d --rm --gpus all --ipc host -v $(pwd):/workspace thesis:latest sleep inf
