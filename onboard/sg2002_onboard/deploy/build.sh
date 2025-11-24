#!/bin/bash
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
set -e

ROOT_PWD=$( cd "$( dirname $0 )" && cd -P "$( dirname "$SOURCE" )" && pwd )

# build
BUILD_DIR=${ROOT_PWD}/build

if [ -z ${GCC_COMPILER} ]
then
  echo "GCC_COMPILER was not set."
  echo "Please set GCC_COMPILER via 'export GCC_COMPILER=~/ws/sg2002/LicheeRV-Nano-Build/host-tools/gcc/riscv64-linux-musl-x86_64/bin'"
  echo "or put this folder under the GCC_COMPILER."
  GCC_COMPILER=$(pwd | sed 's/\(GCC_COMPILER\).*/\1/g')

  STR_MUST_EXIST="GCC_COMPILER"
  if [[ $GCC_COMPILER != *$STR_MUST_EXIST* ]]
  then
    exit
  fi
fi

export PATH=${GCC_COMPILER}:${PATH}

if [ -z ${TPU_SDK_PATH} ]
then
  echo "TPU_SDK_PATH was not set."
  echo "Please set TPU_SDK_PATH via 'export TPU_SDK_PATH=~/ws/sg2002/tpu_sdk/cvitek_tpu_sdk/'"
  echo "or put this folder under the TPU_SDK_PATH."
  TPU_SDK_PATH=$(pwd | sed 's/\(TPU_SDK_PATH\).*/\1/g')

  STR_MUST_EXIST="TPU_SDK_PATH"
  if [[ $TPU_SDK_PATH != *$STR_MUST_EXIST* ]]
  then
    exit
  fi
fi

OPENCV_PATH=${TPU_SDK_PATH}/opencv


if [[ ! -d "${BUILD_DIR}" ]]; then
  mkdir -p ${BUILD_DIR}
fi

PREFIX_DIR=${SCRIPT_DIR}/install

# Build deploy
cd ${BUILD_DIR}
cmake .. \
    -DCMAKE_TOOLCHAIN_FILE=${SCRIPT_DIR}/toolchain-riscv64-linux-musl-x86_64.cmake \
    -DCMAKE_INSTALL_PREFIX=${PREFIX_DIR} \
    -DTPU_SDK_PATH=${TPU_SDK_PATH} \
    -DOPENCV_PATH=${OPENCV_PATH}
make -j$(nproc)
make install
cd -