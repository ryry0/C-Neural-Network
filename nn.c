#include <nn.h>

/*-----------------------------------------------------------------------*/
/*                      MATH FUNCTIONS                                   */
/*-----------------------------------------------------------------------*/
inline float softplus(float z) {
  return log(1.0f + exp(z));
}

inline float softmax(float z) {
  return log(1.0f + exp(z));
}

float sigmoid(float z)  {
  return 1.0f/(1.0f+exp(-z));
}

float sigmoidPrime(float z)  {
  return sigmoid(z)*(1.0f-sigmoid(z));
}

//from Knuth and Marsaglia
float genRandGauss() {
  static double V1, V2, S;
  static int32_t phase = 0;
  float X;

  if(phase == 0) {
    do {
      double U1 = (double)rand() / RAND_MAX;
      double U2 = (double)rand() / RAND_MAX;

      V1 = 2 * U1 - 1;
      V2 = 2 * U2 - 1;
      S = V1 * V1 + V2 * V2;
      } while(S >= 1 || S == 0);

    X = V1 * sqrt(-2 * log(S) / S);
  } else
    X = V2 * sqrt(-2 * log(S) / S);

  phase = 1 - phase;

  return X;
}

/*-----------------------------------------------------------------------*/
/*                            INIT NNET                                  */
/*-----------------------------------------------------------------------*/
bool initNNet(neural_network_t * n_net, size_t num_layers,
    size_t * neurons_per_layer) {

  if (num_layers <= 1)
    return false;

  n_net->num_layers_ = num_layers;

  //allocate array of layers
  n_net->layers_ = (nn_layer_t *) malloc (num_layers * sizeof(nn_layer_t));

  //for each layer
  for (size_t i = 0; i < n_net->num_layers_; i++) {

    nn_layer_t * current_layer = &n_net->layers_[i];

    current_layer->num_neurons_ = neurons_per_layer[i]; //set num neurons

    current_layer->outputs_ = //allocate outputs
      (float *) malloc(current_layer->num_neurons_ * sizeof(float));

    if (i < 1) //skip allocating + initing weights + biases for the first layer
      continue;


    current_layer->errors_ = //allocate errors
      (float *) malloc(current_layer->num_neurons_ * sizeof(float));

    current_layer->avg_errors_ = //allocate avg errors
      (float *) malloc(current_layer->num_neurons_ * sizeof(float));

    current_layer->biases_ = //allocate biases
      (float *) malloc(current_layer->num_neurons_ * sizeof(float));

    current_layer->weighted_sums_ = //allocate weighted sums
      (float *) malloc(current_layer->num_neurons_ * sizeof(float));

    current_layer->weights_ = //allocate weights
      (float **) malloc(current_layer->num_neurons_ * sizeof(float*));


    current_layer->avg_weight_grads_ = //allocate weights
      (float **) malloc(current_layer->num_neurons_ * sizeof(float*));

    current_layer->weights_per_neuron_ = //set weights per neuron
      n_net->layers_[i-1].num_neurons_;     //to num neurons in prev layer


    //for every neuron j in layer allocate and init weights + biases
    for (size_t j = 0; j < current_layer->num_neurons_ ; j++) {
      current_layer->biases_[j] = genRandGauss();

      //num weights depend on size of previous layer
      current_layer->weights_[j] =
        (float *) malloc(current_layer->weights_per_neuron_ *
            sizeof(float));

      current_layer->avg_weight_grads_[j] =
        (float *) malloc(current_layer->weights_per_neuron_ *
            sizeof(float));

      //initialize k weights for the particular neuron, j
      for (size_t k = 0; k < current_layer->weights_per_neuron_; k++) {
        current_layer->weights_[j][k] = genRandGauss();
      }
    } //end for each neuron
  } //end for each layer

  return true;
} //end initNNet

/*-----------------------------------------------------------------------*/
/*                            DESTROY NNET                               */
/*-----------------------------------------------------------------------*/
//frees the allocated nodes
bool destroyNNet(neural_network_t* n_net) {
  if (n_net == NULL)
    return false;

  for (size_t i = 0; i < n_net->num_layers_; i++) {
    free(n_net->layers_[i].outputs_); //free array of biases

    if (i < 1) //no need to free biases and weights for first layer
      continue;

    free(n_net->layers_[i].biases_); //free array of biases
    free(n_net->layers_[i].errors_); //free array of errors_
    free(n_net->layers_[i].avg_errors_); //free array of errors_
    free(n_net->layers_[i].weighted_sums_); //free array of errors_

    for (size_t j = 0; j < n_net->layers_[i].num_neurons_ ; j++) {
          free(n_net->layers_[i].weights_[j]); //free array of weights
          free(n_net->layers_[i].avg_weight_grads_[j]); //free weight gradients
    } //end for each neuron

    free(n_net->layers_[i].weights_); //free array of weight arrays
    free(n_net->layers_[i].avg_weight_grads_);
  } //end for each layer

  free(n_net->layers_); //free array of layers

  return true;
} //end destroyNNet

/*-----------------------------------------------------------------------*/
/*                         CLEAR BATCH AVERAGES                          */
/*-----------------------------------------------------------------------*/
//utility function that clears out avg_weight_grads_ and avg_errors_
void clearBatchAvg(neural_network_t* n_net) {

  for (size_t i = 1; i < n_net->num_layers_; i++) {
    nn_layer_t * current_layer = &n_net->layers_[i];

    memset(current_layer->avg_errors_, 0, current_layer->num_neurons_ *
        sizeof(float));

    for(size_t j = 0; j < current_layer->num_neurons_; j++) {
      memset(current_layer->avg_weight_grads_[j], 0,
          current_layer->weights_per_neuron_ * sizeof(float));
    }
  }
}


/*-----------------------------------------------------------------------*/
/*                      STOCHASTIC GRADIENT DESCENT                      */
/*-----------------------------------------------------------------------*/
//applies stochastic gradient descent on the network.
//SO INEFFICIENT ;-----;
bool sgdNNet(neural_network_t* n_net,
    float* const samples,
    float* const expected,
    size_t num_samples,
    uint64_t epochs,
    float eta,
    size_t mini_batch_size,
    float* verif_samples,     //set of things to classify
    float* verif_expected,  //set of things to compare against
    size_t num_verif_samples) {

  clock_t start, end;
  float cpu_time;

  if (mini_batch_size > num_samples)
    return false;

  printf("Beginning Training\n");

  for(uint64_t i = 0; i < epochs; i++) {
    start = clock();

    //clear the average values for the gradients.
    clearBatchAvg(n_net);

    for (size_t j = 0; j < mini_batch_size; j++) {
      size_t sample_index = rand() % num_samples;

      float* current_sample = //get random sample index
        samples+(n_net->layers_[0].num_neurons_ * sample_index);

      float* current_expected = //get random sample index
        expected+(n_net->layers_[n_net->num_layers_ -1].num_neurons_ *
          sample_index);


      //run backprop alg on the sample and calculate deltas
      backPropNNet(n_net, current_sample, current_expected);

      //calculate avg of partial wks and partial b
      for (size_t k = 1; k < n_net->num_layers_; k++) {
        nn_layer_t * current_layer = &n_net->layers_[k];
        nn_layer_t * prev_layer = &n_net->layers_[k-1];

        for (size_t n = 0; n < current_layer->num_neurons_; n++) {

          current_layer->avg_errors_[n] +=
            current_layer->errors_[n];

          for (size_t m = 0; m < current_layer->weights_per_neuron_; m++) {
            current_layer->avg_weight_grads_[n][m] +=
              (current_layer->errors_[n] * prev_layer->outputs_[m]);
          }
        } //end for neurons
      } //end for each layer

    } //end for mini batch

    //perform gradient descent on the biases and the weights
    for (size_t k = 1; k < n_net->num_layers_; k++) {
      nn_layer_t * current_layer = &n_net->layers_[k];

      for (size_t n = 0; n < current_layer->num_neurons_; n++) {

        current_layer->biases_[n] -= (eta/(float)mini_batch_size) *
          current_layer->avg_errors_[n];

        for (size_t m = 0; m < current_layer->weights_per_neuron_; m++) {
          current_layer->weights_[n][m] -= (eta/(float)mini_batch_size) *
            current_layer->avg_weight_grads_[n][m];
        }
      }
    } //end for each layer


    if ((verif_samples != NULL) && (verif_expected != NULL))
      verifyNNet(n_net, verif_samples, verif_expected, num_verif_samples);

    end = clock();
    cpu_time += ((float) (end - start))/CLOCKS_PER_SEC; //
  } //end for epochs

  printf("Total time: %f seconds.\n", cpu_time);
  printf("Epoch average completion time %f seconds.\n\n", cpu_time/(double)epochs);

  return true;
}

/*-----------------------------------------------------------------------*/
/*                          BACKPROPAGATION                              */
/*-----------------------------------------------------------------------*/

bool backPropNNet(neural_network_t* n_net, float* const input,
    float* const expected) {

  size_t output_layer = n_net->num_layers_ - 1;
  nn_layer_t * current_layer = NULL;
  nn_layer_t * next_layer = NULL;

  //feedforward
  feedForwardNNet(n_net, input);

  //calculate errors for output layer per neuron
  current_layer = &n_net->layers_[output_layer];
  for(size_t i = 0; i < current_layer->num_neurons_; i++) {
    current_layer->errors_[i] =
      (current_layer->outputs_[i] - expected[i]) *
      sigmoidPrime(current_layer->weighted_sums_[i]);
  } //(a - y) * s(z) forall neurons

  //backpropagate the errors in (num_layers_ - 2) to layer 1
  for (size_t i = output_layer - 1; i > 0; i--) {
    current_layer = &n_net->layers_[i];
    next_layer = &n_net->layers_[i+1];

    for(size_t j = 0; j < current_layer->num_neurons_; j++) {
      float dot_product = 0;

      //dot product next layer deltas with their weights
      for(size_t k = 0; k < next_layer->num_neurons_; k++) {
        dot_product += next_layer->weights_[k][j] * next_layer->errors_[k];
      }

      current_layer->errors_[j] = (dot_product) *
        sigmoidPrime(current_layer->weighted_sums_[j]);

    }
  } //end for each layer

  return true;
} //end backProp

/*-----------------------------------------------------------------------*/
/*                            FEEDFORWARD                                */
/*-----------------------------------------------------------------------*/
//feedforward will only take the first layer num_nodes_ worth from data arr
//classification will be returned in the final output layer
void feedForwardNNet(neural_network_t* n_net, float* const input) {

  nn_layer_t * first_layer = &n_net->layers_[0];

  //assign data to first layer of network
  for (size_t i = 0; i < first_layer->num_neurons_; i++) {
    first_layer->outputs_[i] = input[i];
  }

  //optimize here sse/threads
  for (size_t i = 1; i < n_net->num_layers_; i++) { //for each layer
    //optimize this maybe using threads
    nn_layer_t * current_layer = &n_net->layers_[i];
    nn_layer_t * prev_layer = &n_net->layers_[i-1];

    for (size_t j = 0; j < current_layer->num_neurons_; j++) { //for nodes

      //dot product
      float dot_product = 0;

      //since trivial use simd extensions
      for (size_t k = 0; k < current_layer->weights_per_neuron_; k++) {
        dot_product += current_layer->weights_[j][k] *
          prev_layer->outputs_[k];
      }

      //calculate weighted sum
      current_layer->weighted_sums_[j] = dot_product +
        current_layer->biases_[j];

      //calculate neuron j output
      current_layer->outputs_[j] =
        sigmoid(current_layer->weighted_sums_[j]);
    }
  } //end for each layer
} //end feedForwardNNet

/*-----------------------------------------------------------------------*/
/*                                 VERIFY                                */
/*-----------------------------------------------------------------------*/
//will run classification over whole verification data set and print the
//identification rate
void verifyNNet(neural_network_t* n_net,
    float* const input_data,
    float* const expected_data,
    size_t data_size) {

  nn_layer_t * first_layer = &n_net->layers_[0];
  nn_layer_t * output_layer = &n_net->layers_[n_net->num_layers_ -1];
  size_t num_correct = 0;

  for (size_t sample_index = 0; sample_index < data_size; sample_index++) {

    feedForwardNNet(n_net,
        input_data + (sample_index * first_layer->num_neurons_));

    if (getmax(output_layer->outputs_, output_layer->num_neurons_) ==
        getmax(expected_data + (sample_index*output_layer->num_neurons_),
          output_layer->num_neurons_))
      num_correct++;
  }
  printf("Identified %ld correctly out of %ld.\n", num_correct, data_size);
  printf("%f %% success rate\n", ((float) num_correct/ (float)data_size)*100.0);
} //end feedForwardNNet
