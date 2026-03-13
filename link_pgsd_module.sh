#!/bin/bash

git submodule foreach --recursive git checkout main 

export GIT_SRC=$(pwd)
ln -s ${GIT_SRC}/dependencies/pgsd-sph/pgsd/pgsd/pgsd.c ${GIT_SRC}/hoomd-blue/hoomd/extern/pgsd.c
ln -s ${GIT_SRC}/dependencies/pgsd-sph/pgsd/pgsd/pgsd.h ${GIT_SRC}/hoomd-blue/hoomd/extern/pgsd.h
