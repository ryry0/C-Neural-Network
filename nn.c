#include <nn.h>

/*-----------------------------------------------------------------------*/
/*                      MATH FUNCTIONS                                   */
/*-----------------------------------------------------------------------*/
inline double softplus(double z) {
  return log(1.0 + exp(z));
}

inline double softmax(double z) {
  return log(1.0 + exp(z));
}

static inline double sigmoid(double z)  {
  return 1.0/(1.0+exp(-z));
}

static inline double sigmoidPrime(double z)  {
  return sigmoid(z)*(1-sigmoid(z));
}


//from Knuth and Marsaglia
double genRandGauss() {
  static double V1, V2, S;
  static int phase = 0;
  double X;

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

    n_net->layers_[i].num_neurons_ = neurons_per_layer[i]; //set num neurons

    n_net->layers_[i].outputs_ = //allocate outputs
      (double *) malloc(neurons_per_layer[i] * sizeof(double));

    if (i < 1) //skip allocating + initing weights + biases for the first layer
      continue;

    n_net->layers_[i].biases_ = //allocate biases
      (double *) malloc(neurons_per_layer[i] * sizeof(double));

    n_net->layers_[i].weights_ = //allocate weights
      (double **) malloc(neurons_per_layer[i] * sizeof(double*));

    n_net->layers_[i].weights_per_neuron_ = //set weights per neuron
      n_net->layers_[i-1].num_neurons_;     //to num neurons in prev layer


    //for every neuron j in layer allocate and init weights + biases
    for (size_t j = 0; j < n_net->layers_[i].num_neurons_ ; j++) {
      n_net->layers_[i].biases_[j] = genRandGauss();

      //num weights depend on size of previous layer
      n_net->layers_[i].weights_[j] =
        (double *) malloc(n_net->layers_[i].weights_per_neuron_ *
            sizeof(double));

      //initialize k weights for the particular neuron, j
      for (size_t k = 0; k < n_net->layers_[i].weights_per_neuron_; k++) {
        n_net->layers_[i].weights_[j][k] = genRandGauss();
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

    for (size_t j = 0; j < n_net->layers_[i].num_neurons_ ; j++) {
          free(n_net->layers_[i].weights_[j]); //free array of weights
    } //end for each neuron

    free(n_net->layers_[i].weights_); //free array of weight arrays
  } //end for each layer

  free(n_net->layers_); //free array of layers

  return true;
} //end destroyNNet

/*-----------------------------------------------------------------------*/
/*                      STOCHASTIC GRADIENT DESCENT                      */
/*-----------------------------------------------------------------------*/
//applies stochastic gradient descent on the network.
bool sgdNNet(double* const data, neural_network_t* n_net, long epochs,
    double eta, size_t data_size, size_t mini_batch_size) {

  if (mini_batch_size > data_size)
    return false;

  for(long i = 0; i < epochs; i++) {
    //TODO:WILL PROBABLY PRODUCE BAD RESULTS IF DATA_SIZE > RAND_MAX
    for (size_t j = 0; j < mini_batch_size; j++) {

      double* sample = //get random sample index
        data+(n_net->layers_[0].num_neurons_ * (rand() % data_size));

    }
  } //end for epochs

  return true;
}

/*-----------------------------------------------------------------------*/
/*                          BACKPROPAGATION                              */
/*-----------------------------------------------------------------------*/
bool backProp(neural_network_t* n_net) {
  return true;
}

/*-----------------------------------------------------------------------*/
/*                            FEEDFORWARD                                */
/*-----------------------------------------------------------------------*/
//feedforward will only take the first layer num_nodes_ worth from data arr
//classification will be returned in the final output layer
void feedForwardNNet(double* const data, neural_network_t* n_net) {

  //assign data to first layer of network
  for (size_t i = 0; i < n_net->layers_[0].num_neurons_; i++) {
    n_net->layers_[0].outputs_[i] = data[i];
  }

  //optimize here sse/threads
  for (size_t i = 1; i < n_net->num_layers_; i++) { //for each layer
    //optimize this maybe using threads
    for (size_t j = 0; j < n_net->layers_[i].num_neurons_; j++) { //for nodes

      //dot product
      double sum = 0;

      //since trivial use simd extensions
      for (size_t k = 0; k < n_net->layers_[i].weights_per_neuron_; k++) {
        sum += n_net->layers_[i].weights_[j][k] *
          n_net->layers_[i-1].outputs_[k];
      }

      //calculate neuron j output
      n_net->layers_[i].outputs_[j] =
        sigmoid(sum + n_net->layers_[i].biases_[j]);
    }
  }
} //end feedForwardNNet
