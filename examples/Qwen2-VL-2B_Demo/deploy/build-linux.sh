rm -rf build
mkdir build && cd build

GCC_COMPILER_PATH=aarch64-linux-gnu
cmake .. -DCMAKE_CXX_COMPILER=${GCC_COMPILER}/bin/aarch64-none-linux-gnu-g++  \
        -DCMAKE_C_COMPILER=${GCC_COMPILER}/bin/aarch64-none-linux-gnu-gcc \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_SYSTEM_NAME=Linux \
        -DCMAKE_SYSTEM_PROCESSOR=aarch64 \

make -j8
make install