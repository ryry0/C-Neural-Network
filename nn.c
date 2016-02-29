#include <nn.h>


inline double softplus(double z) {
  return log(1.0 + exp(z));
}


inline double softmax(double z) {
  return log(1.0 + exp(z));
}

inline double sigmoid(double z)  {
  return 1.0/(1.0+exp(-z));
}

//refactor code:
//pass random function?
//pass activation function?
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

    n_net->layers_[i].neurons_ =
      (nn_neuron_t *) malloc(neurons_per_layer[i] * sizeof(nn_neuron_t));

    if (i < 1) //skip allocating + initing weights for the first layer
      continue;

    //for every neuron in layer allocate weights
    for (size_t j = 0; j < n_net->layers_[i].num_neurons_ ; j++) {

      //allocate weights depending on number of neurons in prev layer
      //ith layer jth neuron
      size_t num_weights =
      n_net->layers_[i].neurons_[j].num_weights_ = n_net->layers_[i-1].num_neurons_;

      n_net->layers_[i].neurons_[j].weights_ =
        (double *) malloc(n_net->layers_[i-1].num_neurons_ * sizeof(double));

      n_net->layers_[i].neurons_[j].bias_ = genRandGauss();

      //init each weight
      for (size_t k = 0; k < num_weights; k++) {
        n_net->layers_[i].neurons_[j].weights_[k] = genRandGauss();
      } //end for each weight

    } //end for each neuron
  } //end for each layer

  return true;
} //end initNNet

//frees the allocated nodes
bool destroyNNet(neural_network_t* n_net) {
  if (n_net == NULL)
    return false;

  for (size_t i = 0; i < n_net->num_layers_; i++) {
    for (size_t j = 0; j < n_net->layers_[i].num_neurons_ ; j++) {

        if (i < 1) //no need to free weights for first layer
          break;

        free(n_net->layers_[i].neurons_[j].weights_); //free array of weights
    } //end for each neuron

    free(n_net->layers_[i].neurons_); //free array of neurons

  } //end for each layer

  free(n_net->layers_); //free array of layers

  return true;
}

//applies stochastic gradient descent on the network.
bool trainNNet(neural_network_t* n_net, long epochs, size_t layers) {

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
