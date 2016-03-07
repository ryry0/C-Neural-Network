/* TODO:
 * refactor code to be cleaner
 * pass random function? use pcg random function
 * pass activation function?
 * implement simd
 * full matrix based approach (ch2 of nielsen's book)
 */

#ifndef _NN_H_
#define _NN_H_

#include <math.h>    //for math functions
#include <time.h>    //for seeding time
#include <stdbool.h> //for bools
#include <stdlib.h>  //for random
#include <stdint.h>  //for uint*
#include <string.h>
#include <stdio.h>
#include <util.h>
#include <pthread.h>

#define NUM_THREADS 4
typedef struct nn_layer_t {
  size_t num_neurons_;        //num neurons in layer
  size_t weights_per_neuron_; //based on num neurons in prev layer connected

  float* biases_;            //neuron bias
  float* outputs_;           //output of neuron
  float* weighted_sums_;     //weighted sum of neuron z + bias
  float** weights_;          //2d array rows for each neuron

  float* errors_;            //neuron error delta
} nn_layer_t;

typedef struct neural_network_t {
  size_t num_layers_;
  nn_layer_t* layers_;
} neural_network_t;

typedef struct thread_data_t {
  nn_layer_t* current_layer_;
  nn_layer_t* aux_layer_; //auxiliary layer (prev_layer/next_layer);
  size_t start_index_;
  size_t data_size_;
  size_t thr_id;
} thread_data_t;

/*
 * layers: takes num layers inclusive of input and output layers
 * nodes_per_layer: takes array of sizes for each layer. should be arr of size
 *  layers
 * initializes weights to random numbers
 */
bool initNNet(neural_network_t* n_net, size_t num_layers,
    size_t* const nodes_per_layer);

/*
 * Applies stochastic gradient descent on the network.
 * Epochs is how many mini batches to test.
 * num_samples specifies number of samples in input array.
 * Expected output array must be same size of input array.
 * Assumes data is in one long array.
 * This array must be of size data_size * neurons in first layer.
 * Takes the errors calculated in backprop to calculate gradient and thus
 * descent.
 * May want to change this in future to take 2d array
 * Inputs are some multiple of output layer size * num_samples
 * Verification data is used to check progress of epoch training.
 * If the pointer passed is null, the verification step is skipped.
 */
bool sgdNNet(neural_network_t* n_net,
    float* const samples,
    float* const expected,
    size_t num_samples,
    uint64_t epochs,
    float eta ,
    size_t mini_batch_size,
    float* verif_samples,     //set of things to classify
    float* verif_expected,  //set of things to compare against
    size_t num_verif_samples);

/*
 * Given an input and expected output, (and cost function?)
 * calculates the errors in the neural network per node.
 * Should technically output the gradient, and not just the errors, but
 * that's a lot more to store.
 */
bool backPropNNet(neural_network_t* n_net, float* const input,
    float* const expected);

/*
 * Verifies the n_net against the verification data
 */
void verifyNNet(neural_network_t* n_net,
    float* const input_data,
    float* const expected_data,
    size_t data_size);

//runs net input -> output for classification
void feedForwardNNet(neural_network_t* n_net, float* const input);

//calculates the outputs per layer. This is a thread function
void* calcLayerOutputs(void* arguments);

//calculates the errors per layer. This is a thread function
void* calcLayerErrors(void *arguments);

//distributes the calculations for a layer across threads
void distributeCalcs(nn_layer_t* current_layer, nn_layer_t* aux_layer, size_t layer_num,
    void* (*calc_func)(void*));

bool destroyNNet(neural_network_t* n_net);

//utility function that clears out avg_weight_grads_ and avg_errors_
void clearBatchAvg(neural_network_t* n_net);

//from knuth and marsaglia
float genRandGauss();


#endif
