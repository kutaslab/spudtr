#!/bin/bash

# Intended for TravisCI deploy stage where there is exactly one file that matches
#
# ${CONDA_PREFIX}/conda-bld/linux-65/sputdr-*.tar.bz
 
# For testing locally in an active conda env, can fake the TravisCI env like so
#   export TRAVIS="true"
#   export TRAVIS_BRANCH="X.Y.Z" 
#   export ANACONDA_TOKEN="ku-actual-token ..."
#

if [ $USER = "travis" ]; then
    bld_prefix="/home/travis/miniconda"
else
    bld_prefix=${CONDA_PREFIX}
fi


if [ "$TRAVIS" != "true" ]; then
    echo "This is meant for a TravisCI build, do not run locally."
    exit -1
fi

# master branch builds get labeled for anacoda package "main"
# others labeled by branch for testing
if [ $TRAVIS_BRANCH = "master" ]; 
then
    # do *NOT* force master onto main, we want version collisions to fail
    FORCE="" 
    conda_label="main"
else
    # *DO* force non-master braches onto their label so we can conda install latest
    # for testing. Careful, this means non-master branches badly versioned as X.Y.Z 
    # clobber main X.Y.Z
    FORCE="--force"  # 
    conda_label=latest$TRAVIS_BRANCH
fi

# thus far ...
echo "package name: $PACKAGE_NAME"
echo "conda prefix: $CONDA_PREFIX"
echo "travis branch: $TRAVIS_BRANCH"
echo "conda label: $conda_label"
echo "force flag: $FORCE"

# not needed for travis but tar.bz2 files can accumulate during local testing
bz_ls="/bin/ls -1 ${bld_prefix}/conda-bld/linux-64/${PACKAGE_NAME}-*.tar.bz2"
$bz_ls
n_bz2=`${bz_ls} | wc -l`
if (( $n_bz2 != 1 )); then
    echo "found ${n_bz2} ${bld_prefix}/conda-bld/linux-64/${PACKAGE_NAME}"'-*.tar.bz2'
    echo "there must be exactly one to convert for the conda upload"
    exit -2
fi

# force convert even thouhg compiled C extension ... whatever works cross-platform works
rm -f -r ./tmp-conda-builds
mkdir -p ./tmp-conda-builds/linux-64
cp ${bld_prefix}/conda-bld/linux-64/${PACKAGE_NAME}-*.tar.bz2 ./tmp-conda-builds/linux-64
conda convert --platform all ${bld_prefix}/conda-bld/linux-64/${PACKAGE_NAME}-*.tar.bz2 --output-dir ./tmp-conda-builds --force
/bin/ls -l ./tmp-conda-builds/**/${PACKAGE_NAME}-*.tar.bz2

echo "Deploying to Anaconda.org like so ..."
conda_cmd="anaconda --token $ANACONDA_TOKEN upload ./tmp-conda-builds/**/${PACKAGE_NAME}-*.tar.bz2 --label $conda_label --register ${FORCE}"
echo ${conda_cmd}

if ${conda_cmd};
then
    echo "Successfully deployed to Anaconda.org."
else
    echo "Error deploying to Anaconda.org"
    exit -3
fi
exit 0
