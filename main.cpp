#include <iostream>

#include "src/lenet.h"
#include "src/mnist.h"
#include "src/layer/cuda/cuda_utils.h"
#include "src/layer/cuda/cuda_kernel.h"

int main()
{
    cuda_utils agent;
    agent.print_device_info();
    // should give absolute path

    const std::string fashion_mnist_directory = "../data/fashion mnist/";
    const std::string parameter_filepath = "../parameter/weight.bin";
    // load fashion mnist dataset
    MNIST dataset(fashion_mnist_directory);
    dataset.read();
    std::cout << "\nFashion mnist test number: " << dataset.test_labels.cols() << std::endl
              << std::endl;

    std::chrono::_V2::system_clock::time_point start, end;
    std::chrono::nanoseconds elapsed;
    std::cout << "<--------------CPU version-------------->" << std::endl;
    // create network
    Network dnn = create_lenet5_network(parameter_filepath);

    start = std::chrono::high_resolution_clock::now();
    dnn.forward(dataset.test_data);
    end = std::chrono::high_resolution_clock::now();
    elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    std::cout << "CPU total forward time: " << elapsed.count() / 1000000000.0 << "s" << std::endl;
    float acc = compute_accuracy(dnn.output(), dataset.test_labels);
    std::cout << "CPU accuracy: " << acc << std::endl
              << std::endl;

    std::cout << "<--------------Naive GPU version-------------->" << std::endl;
    Network dnn_gpu = create_lenet5_network_gpu(parameter_filepath);

    start = std::chrono::high_resolution_clock::now();
    dnn_gpu.forward(dataset.test_data);
    end = std::chrono::high_resolution_clock::now();
    elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    std::cout << "Naive GPU total forward time: " << elapsed.count() / 1000000000.0 << "s" << std::endl;
    float acc_gpu = compute_accuracy(dnn_gpu.output(), dataset.test_labels);
    std::cout << "Naive GPU accuracy: " << acc_gpu << std::endl
              << std::endl;

    std::cout << "<--------------SMEM GPU version-------------->" << std::endl;
    Network shared_dnn = create_lenet5_network_gpu(parameter_filepath, cuda_kernel::shared);

    start = std::chrono::high_resolution_clock::now();
    shared_dnn.forward(dataset.test_data);
    end = std::chrono::high_resolution_clock::now();
    elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    std::cout << "Shared Memory GPU total forward time: " << elapsed.count() / 1000000000.0 << "s" << std::endl;
    float acc_gpu_shared = compute_accuracy(shared_dnn.output(), dataset.test_labels);
    std::cout << "Shared Memory GPU accuracy: " << acc_gpu_shared << std::endl
              << std::endl;

    std::cout << "<--------------SMEM + CMEM GPU version-------------->" << std::endl;
    Network shared_const_dnn = create_lenet5_network_gpu(parameter_filepath, cuda_kernel::shared_constmem);

    start = std::chrono::high_resolution_clock::now();
    shared_const_dnn.forward(dataset.test_data);
    end = std::chrono::high_resolution_clock::now();
    elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    std::cout << "Shared Memory & Constant Memory GPU total forward time: " << elapsed.count() / 1000000000.0 << "s" << std::endl;
    float acc_gpu_shared_const = compute_accuracy(shared_const_dnn.output(), dataset.test_labels);
    std::cout << "Shared Memory & Constant Memory GPU accuracy: " << acc_gpu_shared_const << std::endl
              << std::endl;
}