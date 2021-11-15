#!/bin/bash


cd ../../pytorch #where I have my `git clone` of pytorch

export ANDROID_ABI=arm64-v8a

BUILD_PYTORCH_MOBILE=1 ANDROID_ABI=arm64-v8a ./scripts/build_android.sh -DBUILD_BINARY=ON
