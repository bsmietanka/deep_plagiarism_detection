#!/bin/bash

docker run -d --rm --gpus all --ipc host -p 8765:8765 -v $(pwd):/workspace thesis:latest sleep inf
