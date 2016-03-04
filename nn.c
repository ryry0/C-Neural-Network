#include <nn.h>

/*-----------------------------------------------------------------------*/
/*                      MATH FUNCTIONS                                   */
/*-----------------------------------------------------------------------*/
/*
inline void softplus(mpfr_t* z, mpfr_t* output) {
  return log(1.0f + exp(z));
}

inline void softmax(mpfr_t* z, mpfr_t* output) {
  return log(1.0f + exp(z));
}
*/

//swapped arguments to match with mpfr architeture
void sigmoid(mpfr_t* output, mpfr_t* const z)  {

  mpfr_t temp;
  mpfr_init2(temp, PRECISION);

  mpfr_neg(temp, *z, MPFR_RNDN);                 //temp = -z
  mpfr_exp(*output, temp, MPFR_RNDN);            //output = exp(temp)
  mpfr_add_d(*output, *output, 1.0f, MPFR_RNDN); //output = 1 + output
  mpfr_d_div(*output, 1.0f, *output, MPFR_RNDN); //output = 1/output

  //output = 1/(1+exp(-z))
  mpfr_clear(temp);
}

void sigmoidPrime(mpfr_t* output, mpfr_t* const z)  {
  mpfr_t temp;
  mpfr_init2(temp, PRECISION);

  sigmoid(&temp, z);  //temp = sigmoid(z)

  mpfr_d_sub(*output, 1.0f, temp, MPFR_RNDN); //output= 1-temp = 1-sigmoid(z)
  mpfr_mul(*output, temp, *output, MPFR_RNDN);//output = sigmoid(z)*output
  //output = sigmoid(z)*(1-sigmoid(z))

  mpfr_clear(temp);
}

//from Knuth and Marsaglia
double genRandGauss() {
  static double V1, V2, S;
  static int32_t phase = 0;
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

    nn_layer_t* current_layer = &n_net->layers_[i];

    current_layer->num_neurons_ = neurons_per_layer[i]; //set num neurons

    current_layer->outputs_ = //allocate outputs
      (mpfr_t *) malloc(current_layer->num_neurons_ * sizeof(mpfr_t));

    //initialize output mpfr variables
    for (size_t j = 0; j < current_layer->num_neurons_; ++j) {
      mpfr_init2(current_layer->outputs_[i], PRECISION);
      mpfr_set_ui(current_layer->outputs_[i], 0);
    }

    if (i < 1) //skip allocating + initing weights + biases for the first layer
      continue;

    current_layer->weights_per_neuron_ = //set weights per neuron
      n_net->layers_[i-1].num_neurons_;     //to num neurons in prev layer

    current_layer->errors_ = //allocate errors
      (mpfr_t *) malloc(current_layer->num_neurons_ * sizeof(mpfr_t));

    current_layer->sum_errors_ = //allocate avg errors
      (mpfr_t *) malloc(current_layer->num_neurons_ * sizeof(mpfr_t));

    current_layer->biases_ = //allocate biases
      (mpfr_t *) malloc(current_layer->num_neurons_ * sizeof(mpfr_t));

    current_layer->weighted_sums_ = //allocate weighted sums
      (mpfr_t *) malloc(current_layer->num_neurons_ * sizeof(mpfr_t));

    current_layer->weights_ = //allocate weights
      (mpfr_t **) malloc(current_layer->num_neurons_ * sizeof(mpfr_t*));


    current_layer->sum_weight_grads_ = //allocate weights
      (mpfr_t **) malloc(current_layer->num_neurons_ * sizeof(mpfr_t*));

    //for every neuron j in layer allocate and init weights + biases
    for (size_t j = 0; j < current_layer->num_neurons_ ; j++) {

      //initialize mpfr variables
      mpfr_init2(current_layer->biases_[j], PRECISION);
      mpfr_set_d(current_layer->biases_[j], genRandGauss(), MPFR_RNDN);

      mpfr_init2(current_layer->weighted_sums_[j], PRECISION);
      mpfr_set_ui(current_layer->weighted_sums_[j], 0, MPFR_RNDN);

      mpfr_init2(current_layer->errors_[j], PRECISION);
      mpfr_set_ui(current_layer->errors_[j], 0, MPFR_RNDN);

      mpfr_init2(current_layer->sum_errors_[j], PRECISION);
      mpfr_set_ui(current_layer->sum_errors_[j], 0, MPFR_RNDN);

      //num weights depend on size of previous layer
      current_layer->weights_[j] =
        (mpfr_t *) malloc(current_layer->weights_per_neuron_ *
            sizeof(mpfr_t));

      current_layer->sum_weight_grads_[j] =
        (mpfr_t *) malloc(current_layer->weights_per_neuron_ *
            sizeof(mpfr_t));

      //initialize k weights for the particular neuron, j
      for (size_t k = 0; k < current_layer->weights_per_neuron_; k++) {
        mpfr_init2(current_layer->sum_weight_grads_[j][k], PRECISION);
        mpfr_set_ui(current_layer->sum_weight_grads_[j][k], 0, MPFR_RNDN);

        mpfr_init2(current_layer->weights_[j][k], PRECISION);
        mpfr_set_d(current_layer->weights_[j][k], genRandGauss(), MPFR_RNDN);
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
    nn_layer_t* current_layer = &n_net->layers_[i];

    for (size_t j = 0; j < current_layer->num_neurons_ ; j++) {
      //clear mpfr variables
      mpfr_clear(current_layer->outputs_[j]);

      if (i < 1) //no need to free biases and weights for first layer
        continue;

      mpfr_clear(current_layer->errors_[j]);
      mpfr_clear(current_layer->sum_errors_[j]);
      mpfr_clear(current_layer->biases_[j]);
      mpfr_clear(current_layer->weighted_sums_[j]);

      for (size_t k = 0; k < current_layer->weights_per_neuron_; k++) {
        mpfr_clear(current_layer->weights_[j][k]);
        mpfr_clear(current_layer->sum_weight_grads_[j][k]);
      }

      free(n_net->layers_[i].weights_[j]); //free array of weights
      free(n_net->layers_[i].sum_weight_grads_[j]); //free weight gradients
    } //end for each neuron

    free(current_layer->outputs_); //free array of biases

    if (i < 1) //no need to free biases and weights for first layer
      continue;

    free(current_layer->biases_); //free array of biases
    free(current_layer->errors_); //free array of errors_
    free(current_layer->sum_errors_); //free array of errors_
    free(current_layer->weighted_sums_); //free array of errors_

    free(current_layer->weights_); //free array of weight arrays
    free(current_layer->sum_weight_grads_);
  } //end for each layer

  free(n_net->layers_); //free array of layers

  return true;
} //end destroyNNet

/*-----------------------------------------------------------------------*/
/*                         CLEAR BATCH AVERAGES                          */
/*-----------------------------------------------------------------------*/
//utility function that clears out sum_weight_grads_ and sum_errors_
void clearBatchAvg(neural_network_t* n_net) {
  for (size_t i = 1; i < n_net->num_layers_; i++) {
    nn_layer_t * current_layer = &n_net->layers_[i];

    for(size_t j = 0; j < current_layer->num_neurons_; j++) {
      mpfr_set_ui(current_layer->sum_errors_[j], 0, MPFR_RNDN);
      for(size_t k = 0; k < current_layer->weights_per_neuron_; k++) {
        mpfr_set_ui(current_layer->sum_weight_grads_[j][k], 0, MPFR_RNDN);
      }
    }
  }
} //end clearBatchAvg


/*-----------------------------------------------------------------------*/
/*                      STOCHASTIC GRADIENT DESCENT                      */
/*-----------------------------------------------------------------------*/
//applies stochastic gradient descent on the network.
//SO INEFFICIENT ;-----;
bool sgdNNet(neural_network_t* n_net,
    mpfr_t* const samples,
    mpfr_t* const expected,
    size_t num_samples,
    uint64_t epochs,
    double eta,
    size_t mini_batch_size,
    mpfr_t* verif_samples,     //set of things to classify
    mpfr_t* verif_expected,  //set of things to compare against
    size_t num_verif_samples) {

  clock_t start, end;
  double cpu_time;
  mpfr_t temp;
  mpfr_t learning_const;
  mpfr_init2(temp, PRECISION);
  mpfr_init2(learning_const, PRECISION);
  //learning const = eta / mini_batch_size
  mpfr_set_d(learning_const, eta/(double)mini_batch_size, MPFR_RNDN);

  if (mini_batch_size > num_samples)
    return false;

  for(uint64_t i = 0; i < epochs; i++) {
    //TODO:WILL PROBABLY PRODUCE BAD RESULTS IF DATA_SIZE > RAND_MAX
    //might get same random number

    printf("Epoch %ld of %ld\n", i+1, epochs);
    start = clock();

    //clear the average values for the gradients.
    clearBatchAvg(n_net);

    for (size_t j = 0; j < mini_batch_size; j++) {
      size_t sample_index = rand() % num_samples;

      printf("Batch %ld of %ld\n", j+1, mini_batch_size);

      //printf("rand %ld: %ld\n", j, sample_index);
      mpfr_t* current_sample = //get random sample index
        samples+(n_net->layers_[0].num_neurons_ * sample_index);

      mpfr_t* current_expected = //get random sample index
        expected+(n_net->layers_[n_net->num_layers_ -1].num_neurons_ *
          sample_index);


      //run backprop alg on the sample and calculate deltas
      backPropNNet(n_net, current_sample, current_expected);

      //calculate avg of partial wks and partial b
      for (size_t k = 1; k < n_net->num_layers_; k++) {
        nn_layer_t * current_layer = &n_net->layers_[k];
        nn_layer_t * prev_layer = &n_net->layers_[k-1];

        for (size_t n = 0; n < current_layer->num_neurons_; n++) {

          //sum_errors_[n] = sum_errors_[n] + errors_[n]
          mpfr_add(current_layer->sum_errors_[n],
              current_layer->sum_errors_[n],
              current_layer->errors_[n],
              MPFR_RNDN);

          for (size_t m = 0; m < current_layer->weights_per_neuron_; m++) {
            //temp = errors_[n] * p_outputs_[m]
            mpfr_mul(temp, current_layer->errors_[n], prev_layer->outputs_[m],
                MPFR_RNDN);

            //sum_weight_grads_[n][m] = sum_weight_grads_[n][m] + temp
            mpfr_add(current_layer->sum_weight_grads_[n][m],
                current_layer->sum_weight_grads_[n][m],
                temp,
                MPFR_RNDN);
          }
        } //end for neurons
      } //end for each layer

    } //end for mini batch

    //perform gradient descent on the biases and the weights
    for (size_t k = 1; k < n_net->num_layers_; k++) {
      nn_layer_t * current_layer = &n_net->layers_[k];

      for (size_t n = 0; n < current_layer->num_neurons_; n++) {

        //temp = learning_const * sum_errors_[n]
        mpfr_mul(temp, learning_const, current_layer->sum_errors_[n]);

        //biases_[n] = biases_[n] - temp i.e
        //biases_[n] = biases_[n] - (learning_const * sum_errors_[n])
        mpfr_sub(current_layer->biases_[n],
            current_layer->biases_[n],
            temp,
            MPFR_RNDN);

        for (size_t m = 0; m < current_layer->weights_per_neuron_; m++) {
          //temp = learning_const * sum_weight_grads_
          mpfr_mul(temp, learning_const,
              current_layer->sum_weight_grads_[n][m],
              MPFR_RNDN);

          //weights_[n][m] = weights_[n][m] - (learning_const*sum_weight_grads_[n][m])
          mpfr_sub(current_layer->weights_[n][m],
              current_layer->weights_[n][m],
              temp,
              MPFR_RNDN);

          current_layer->weights_[n][m] -= (eta/(mpfr_t)mini_batch_size) *
            current_layer->sum_weight_grads_[n][m];
        }
      }
    } //end for each layer


    if ((verif_samples != NULL) && (verif_expected != NULL))
      verifyNNet(n_net, verif_samples, verif_expected, num_verif_samples);

    end = clock();
    cpu_time = ((mpfr_t) (end - start))/CLOCKS_PER_SEC; //
    printf("Completed in %f seconds.\n\n", cpu_time);
  } //end for epochs

  mpfr_clear(learning_const);
  mpfr_clear(temp);
  return true;
} //end sgdNNet

/*-----------------------------------------------------------------------*/
/*                          BACKPROPAGATION                              */
/*-----------------------------------------------------------------------*/

bool backPropNNet(neural_network_t* n_net, mpfr_t* const input,
    mpfr_t* const expected) {

  size_t output_layer = n_net->num_layers_ - 1;
  nn_layer_t * current_layer = NULL;
  nn_layer_t * next_layer = NULL;

  //feedforward
  feedForwardNNet(n_net, input);
  //printImage(input, n_net->layers_[0].num_neurons_);
  //printf("\n");

  mpfr_t temp;
  mpfr_t dot_product;
  mpfr_init2(temp, PRECISION);
  mpfr_init2(dot_product, PRECISION);

  //calculate errors for output layer per neuron
  current_layer = &n_net->layers_[output_layer];
  for(size_t i = 0; i < current_layer->num_neurons_; i++) {

    //errors_[i] = outputs_[i] - expected[i]
    mpfr_sub(current_layer->errors_[i],
        current_layer->outputs_[i], expected[i],
        MPFR_RNDN);

    //temp = sigmoid_Prime(weighted_sums_[i])
    sigmoidPrime(temp, &current_layer->weighted_sums_[i]);

    //errors_[i] = errors_[i] * temp, i.e
    //errors_[i] = (outputs_[i] - expected[i])*sigmoid(weighted_sums_[i])
    mpfr_mul(current_layer->errors_[i],
        current_layer->errors_[i],
        temp,
        MPFR_RNDN);

    /*
    printf("E:%f\tO:%f\tX:%f\n", current_layer->errors_[i],
      current_layer->outputs_[i],
      expected[i]);
      */
  } //(a - y) * s(z) forall neurons

  //backpropagate the errors in (num_layers_ - 2) to layer 1
  for (size_t i = output_layer - 1; i > 0; i--) {
    current_layer = &n_net->layers_[i];
    next_layer = &n_net->layers_[i+1];

    for(size_t j = 0; j < current_layer->num_neurons_; j++) {
      mpfr_set_ui(dot_product, 0, MPFR_RNDN);

      //dot product next layer deltas with their weights
      for(size_t k = 0; k < next_layer->num_neurons_; k++) {
        //temp = n_weights[k][j] * n_errors_[k]
        mpfr_mul(temp, next_layer->weights_[k][j],
            next_layer->errors_[k],
            MPFR_RNDN);

        //dot_product = dot_product + temp
        mpfr_add(dot_product, dot_product, temp, MPFR_RNDN);
      }

      //temp = sigmoidPrime(weighted_sums_[j])
      sigmoidPrime(&temp, &current_layer->weighted_sums_[j], MPFR_RNDN);

      //errors_[j] = dot_product * sigmoidPrime (weighted_sums_[j]);
      mpfr_mul(current_layer->errors_[j], dot_product, temp, MPFR_RNDN);

      /*
      printf("E:%f\tO:%f\n", current_layer->errors_[j],
        current_layer->outputs_[j]);
      */
    }
  } //end for each layer

  mpfr_clear(temp);
  mpfr_clear(dot_product);
  return true;
} //end backProp

/*-----------------------------------------------------------------------*/
/*                            FEEDFORWARD                                */
/*-----------------------------------------------------------------------*/
//feedforward will only take the first layer num_nodes_ worth from data arr
//classification will be returned in the final output layer
void feedForwardNNet(neural_network_t* n_net, mpfr_t* const input) {

  nn_layer_t * first_layer = &n_net->layers_[0];

  mpfr_t dot_product;
  mpfr_t product;

  mpfr_init2(dot_product, PRECISION);
  mpfr_init2(product, PRECISION);

  //assign data to first layer of network
  for (size_t i = 0; i < first_layer->num_neurons_; i++) {
    mfpr_set(first_layer->outputs_[i], input[i], MPFR_RNDN);
  }

  //optimize here sse/threads
  for (size_t i = 1; i < n_net->num_layers_; i++) { //for each layer
    //optimize this maybe using threads
    nn_layer_t * current_layer = &n_net->layers_[i];
    nn_layer_t * prev_layer = &n_net->layers_[i-1];

    for (size_t j = 0; j < current_layer->num_neurons_; j++) { //for nodes

      //dot product
      mpfr_set_ui(dot_product, 0, MPFR_RNDN); //dot product = 0

      //since trivial use simd extensions
      for (size_t k = 0; k < current_layer->weights_per_neuron_; k++) {
        mpfr_mul(product, current_layer->weights_[j][k], prev_layer->outputs_[k],
            MPFR_RNDN); //product = weights_[j][k]*p_outputs[k]

        //dot_product = dot_product + product
        mpfr_add(dot_product, dot_product, product, MPFR_RNDN);
      }

      //calculate weighted sum
      //weighted_sums_[j] = dot_product + biases[j]
      mpfr_add(current_layer->weighted_sums_[j], dot_product,
        current_layer->biases_[j], MPFR_RNDN);

      //calculate neuron j output
      //outputs_[j] = sigmoid(weighted_sums_[j])
      sigmoid(&current_layer->outputs_[j],
          &current_layer->weighted_sums_[j]);
    }
  } //end for each layer

  mpfr_clear(dot_product);
  mpfr_clear(product);
} //end feedForwardNNet

/*-----------------------------------------------------------------------*/
/*                                 VERIFY                                */
/*-----------------------------------------------------------------------*/
//will run classification over whole verification data set and print the
//identification rate
void verifyNNet(neural_network_t* n_net,
    mpfr_t* const input_data,
    mpfr_t* const expected_data,
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
