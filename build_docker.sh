#!/bin/bash

git archive -o docker/dsa-fuzz.tar.gz master
tar -cf docker/X86-64-semantics.tar.gz third-party/X86-64-semantics/semantics

cd docker/
docker build -t synthesis:latest .
