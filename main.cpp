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

    const std::string fashion_mnist_directory = "../data/fashion/";
    const std::string parameter_filepath = "../parameter/weight.bin";
    // load fashion mnist dataset
    MNIST dataset(fashion_mnist_directory);
    dataset.read();
    std::cout << "\nFashion mnist test number: " << dataset.test_labels.cols() << std::endl << std::endl;

    GpuTimer timer;
    // std::cout << "**************CPU version**************" << std::endl;
    // // create network
    // Network dnn = create_lenet5_network(parameter_filepath);

    // timer.Start();
    // dnn.forward(dataset.test_data);
    // timer.Stop();
    // std::cout << "CPU forward time: " << timer.Elapsed() / 1000 << " secs" << std::endl;
    // float acc = compute_accuracy(dnn.output(), dataset.test_labels);
    // std::cout << "CPU accuracy: " << acc << std::endl << std::endl;


    // std::cout << "********Multi-stream GPU version*********" << std::endl;
    // bool multi_stream = true;
    // Network multi_stream_dnn = create_lenet5_network_gpu(parameter_filepath, multi_stream, false);

    // timer.Start();
    // multi_stream_dnn.forward(dataset.test_data);
    // timer.Stop();
    // std::cout << "Multi-stream GPU forward time: " << timer.Elapsed() / 1000 << " secs" << std::endl;
    // float acc_gpu_multi_stream = compute_accuracy(multi_stream_dnn.output(), dataset.test_labels);
    // std::cout << "Multi-stream GPU accuracy: " << acc_gpu_multi_stream << std::endl << std::endl;

    // std::cout << "***************GPU version***************" << std::endl;
    // // create network
    // Network dnn_gpu = create_lenet5_network_gpu(parameter_filepath);

    // timer.Start();
    // dnn_gpu.forward(dataset.test_data);
    // timer.Stop();
    // std::cout << "GPU forward time: " << timer.Elapsed() / 1000 << " secs" << std::endl;
    // float acc_gpu = compute_accuracy(dnn_gpu.output(), dataset.test_labels);
    // std::cout << "GPU accuracy: " << acc_gpu << std::endl << std::endl;

    std::cout << "***************SMEM version***************" << std::endl;
    // create network
    bool shared_mem = true;
    Network shared_dnn = create_lenet5_network_gpu(parameter_filepath, false, shared_mem);

    timer.Start();
    shared_dnn.forward(dataset.test_data);
    timer.Stop();
    std::cout << "Multi-stream GPU forward time: " << timer.Elapsed() / 1000 << " secs" << std::endl;
    float acc_gpu_shared = compute_accuracy(shared_dnn.output(), dataset.test_labels);
    std::cout << "Multi-stream GPU accuracy: " << acc_gpu_shared << std::endl << std::endl;



}