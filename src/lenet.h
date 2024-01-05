#ifndef LENET_H
#define LENET_H

/**
 * @file lenet.h
 * @brief Create the lenet 5 network that we will use for this project.
 * Layer 1: 6 kernels with size 5 × 5, use ReLU activation function.
 * Layer 2: Max Pooling with size 2 × 2
 * Layer 3: 16 kernels with size 5 × 5, use ReLU activation function.
 * Layer 4: Max Pooling with size 2 × 2
 * Layer 5: Flatten
 * Layer 6: dense layer with 120 outputs, use ReLU activation function
 * Layer 7: dense layer with 84 outputs, use ReLU activation function
 * Layer 8: dense layer with 10 outputs, use softmax activation function
 */

#include "network.h"
#include "./layer/conv.h"
#include "./layer/max_pooling.h"
#include "./layer/fully_connected.h"
#include "./layer/softmax.h"
#include "./layer/relu.h"
#include "./layer/conv_gpu.h"
/**
 * @brief Create the lenet 5 network that we will use for this project.
 * 
 * @param parameter_filepath filepath to the parameter file
 * @return Network object with layers added (and parameters loaded if parameter_filepath is not empty)
 */
Network create_lenet5_network(const std::string& parameter_filepath = "");
Network create_lenet5_network_gpu(const std::string& parameter_filepath = "");

#endif // LENET_H