#include <iostream>

#include "src/lenet.h"
#include "src/mnist.h"

int main()
{
    // should give absolute path
    const std::string fashion_mnist_directory = "D:\\Source Code\\Visual Studio Code\\mini-dnn-cpp - Copy\\data\\fashion mnist\\";
    const std::string parameter_filepath = "D:\\Source Code\\Visual Studio Code\\mini-dnn-cpp - Copy\\weight.bin";

    // load fashion mnist dataset
    MNIST dataset(fashion_mnist_directory);
    dataset.read();
    std::cout << "fashion mnist test number: " << dataset.test_labels.cols() << std::endl;

    // create network
    Network dnn = create_lenet5_network(parameter_filepath);

    // forward test data
    dnn.forward(dataset.test_data);
    float acc = compute_accuracy(dnn.output(), dataset.test_labels);
    std::cout << "Test accuracy: " << acc << std::endl;
    std::cout << std::endl;
}