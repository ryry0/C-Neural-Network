//Need a better random number generator

#ifndef _NN_H_
#define _NN_H_
#include <math.h>    //for math functions
#include <time.h>    //for seeding time
#include <stdbool.h> //for bools
#include <stdlib.h>  //for random

typedef struct nn_neuron_t {
  double bias_;
  size_t num_weights_;
  double* weights_; //array of input weights depends on # neurons
  double output_;
} nn_neuron_t;

typedef struct nn_layer_t {
  size_t num_neurons_;
  nn_neuron_t* neurons_;
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
bool initNNet(neural_network_t* n_net, size_t num_layers, size_t* nodes_per_layer);

//applies stochastic gradient descent on the network.
bool trainNNet(neural_network_t* n_net, long epochs, size_t layers);

//runs net input -> output for classification
bool feedForwardNNet(neural_network_t* n_net);

bool destroyNNet(neural_network_t* n_net);

//from knuth and marsaglia
double genRandGauss();

#endif
