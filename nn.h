//refactor code:
//pass random function? use pcg random function
//pass activation function?
//implement simd

#ifndef _NN_H_
#define _NN_H_

#include <math.h>    //for math functions
#include <time.h>    //for seeding time
#include <stdbool.h> //for bools
#include <stdlib.h>  //for random

typedef struct nn_layer_t {
  size_t num_neurons_;
  size_t weights_per_neuron_;
  double* biases_;
  double** weights_; //2d array rows for each neuron
  double* outputs_;
} nn_layer_t;

typedef struct neural_network_t {
  size_t num_layers_;
  nn_layer_t* layers_;
} neural_network_t;

/*
 * layers: takes num layers inclusive of input and output layers
 * nodes_per_layer: takes array of sizes for each layer. should be arr of size
 *  layers
 * initializes weights to random numbers
 */
bool initNNet(neural_network_t* n_net, size_t num_layers,
    size_t* const nodes_per_layer);

//applies stochastic gradient descent on the network.
bool trainNNet(neural_network_t* n_net, long epochs);

//runs net input -> output for classification
void feedForwardNNet(double* const data, neural_network_t* n_net);

bool destroyNNet(neural_network_t* n_net);

//from knuth and marsaglia
double genRandGauss();

#endif
