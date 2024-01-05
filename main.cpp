#include <iostream>

#include "src/lenet.h"
#include "src/mnist.h"
#include "src/layer/cuda/helper.h"
#include "src/layer/cuda/cuda_manager.h"

int main()
{
    cuda_helper cuda_helper;
    cuda_helper.print_device_info();
    // should give absolute path
    // const std::string fashion_mnist_directory = "D:\\Source Code\\Visual Studio Code\\mini-dnn-cpp - Copy\\data\\fashion mnist\\";
    // const std::string parameter_filepath = "D:\\Source Code\\Visual Studio Code\\mini-dnn-cpp - Copy\\parameter\\weight.bin";

    const std::string fashion_mnist_directory = "/home/hdnminh/Documents/ltss/final-project/final-project-parallel-programming/mini-dnn-cpp-clone/data/fashion mnist/";
    const std::string parameter_filepath = "/home/hdnminh/Documents/ltss/final-project/final-project-parallel-programming/mini-dnn-cpp-clone/parameter/weight.bin";
    // load fashion mnist dataset
    MNIST dataset(fashion_mnist_directory);
    dataset.read();
    std::cout << "\nFashion mnist test number: " << dataset.test_labels.cols() << std::endl << std::endl;

    std::cout << "**************CPU version**************" << std::endl;
    // create network
    Network dnn = create_lenet5_network(parameter_filepath);
    GpuTimer timer;
    timer.Start();
    dnn.forward(dataset.test_data);
    timer.Stop();
    std::cout << "CPU forward time: " << timer.Elapsed() / 1000 << " secs" << std::endl;
    float acc = compute_accuracy(dnn.output(), dataset.test_labels);
    std::cout << "CPU accuracy: " << acc << std::endl << std::endl;

    std::cout << "**************GPU version**************" << std::endl;

}