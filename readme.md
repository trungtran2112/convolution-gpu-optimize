# Introduction
**convolution-gpu-optimize** is the final project of parallel programming course at Ho Chi Minh University of Science.
- Our project uses [mini-dnn-cpp](https://github.com/iamhankai/mini-dnn-cpp) as the base project and implements some GPU optimization to the Convolution layer.
- The project uses the newer, modified version of LeNet-5 which contains these layers:
    - Layer 1: 6 kernels with size 5 × 5, use ReLU activation function.
    - Layer 2: Max Pooling with size 2 × 2
    - Layer 3: 16 kernels with size 5 × 5, use ReLU activation function.
    - Layer 4: Max Pooling with size 2 × 2
    - Layer 5: Flatten
    - Layer 6: dense layer with 120 outputs, use ReLU activation function
    - Layer 7: dense layer with 84 outputs, use ReLU activation function
    - Layer 8: dense layer with 10 outputs, use softmax activation function

## Usage
- Download and unzip Fashion MNIST dataset and move to `convolution-gpu-optimize/data/fashion mnist/`
- We have had the LeNet-5 model for the Fashion MNIST dataset trained and saved in [weight.bin](./parameter/weight.bin).<br>
  You can use the provided file or use your own (remember to change the file path in [main.cpp](./main.cpp))
- The below instruction is for Linux-based system, Windows system may be different 
```shell
mkdir build
cd build
cmake ..
make
```
- Run 
```shell
cd build
./main
```