#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <nn.h>

#define NUM_LAYERS 4
#define INPUT_LAYER_SIZE  784 //have to make sure you can map data to inputs
#define OUTPUT_LAYER_SIZE 10

#define PIC_WIDTH 28
#define PIC_HEIGHT 28
#define PICTURE_SIZE 784 //num pixels 28*28

#define NUM_PICTURES 60000
#define NUM_SAMPLES 50000
#define VERIF_SAMPLES 10000

#define NUM_EPOCHS 1000
#define ETA 3.0
#define BATCH_SIZE 10

#define TRAIN_OFFSET      0x10
#define TRAIN_EXP_OFFSET  0x08

#define DEFAULT_TRAIN     "data/train-images-idx3-ubyte"
#define DEFAULT_EXPECT    "data/train-labels-idx1-ubyte"

void classify(neural_network_t *n_net, float* const input_data);

int main(int argc, char ** argv) {
  neural_network_t neural_net;
  size_t layer_sizes[NUM_LAYERS] = {INPUT_LAYER_SIZE,800,
    800,OUTPUT_LAYER_SIZE};
  //load data

  int input_data_fd = 0;
  int expected_data_fd = 0;

  if (argc > 2) {
    input_data_fd = open(argv[1], O_RDONLY);
    expected_data_fd = open(argv[2], O_RDONLY);
  }
  else {
    input_data_fd = open(DEFAULT_TRAIN, O_RDONLY);
    expected_data_fd = open(DEFAULT_EXPECT, O_RDONLY);
  }

  if ((expected_data_fd == -1) || (input_data_fd == -1)) {
    printf("Please provide input data and verification data\n");
    return 1;
  }

  float* input_data = (float *)
    malloc(NUM_PICTURES*PICTURE_SIZE*sizeof(float));

  float* expected_data = (float *)
    calloc(NUM_PICTURES*OUTPUT_LAYER_SIZE,sizeof(float));

  //for MNIST data
  //set input data to first input
  //set output data to first output value
  //should probably mmap the file or something
  lseek(input_data_fd, TRAIN_OFFSET, SEEK_SET);
  lseek(expected_data_fd, TRAIN_EXP_OFFSET, SEEK_SET);

  printf("Copying input data.\n");
  for (size_t i = 0; i < NUM_PICTURES*PICTURE_SIZE; i++) {
    uint8_t buff = 0;
    read(input_data_fd, &buff, 1);
    input_data[i] = ((float) buff/255.0f);
  }

  printf("Copying expected data and mapping it to vectors.\n");
  for (size_t i = 0; i < NUM_PICTURES; i++) {
    uint8_t buff = 0;
    read(expected_data_fd, &buff, 1);
    expected_data[(i*OUTPUT_LAYER_SIZE) + (size_t) buff] = 1.0f;
  }

  /*------------------------------------------------------------------------*/
  /*                      Training the neural net                           */
  /*------------------------------------------------------------------------*/
  srand(time(NULL));
  initNNet(&neural_net, NUM_LAYERS, layer_sizes);

  //train neural net
  sgdNNet(&neural_net,  //n_net
      input_data,       //input
      expected_data,    //expected
      NUM_SAMPLES,      //#samples in data
      NUM_EPOCHS,               //epochs
      ETA,              //eta
      BATCH_SIZE,               //batch size
      NULL,//input_data+NUM_SAMPLES*PICTURE_SIZE, verification input data
      NULL,//expected_data+NUM_SAMPLES*OUTPUT_LAYER_SIZE, verif expected data
      VERIF_SAMPLES);               //verification sample size

  printf("Verifying Neural Net\n");
  for(size_t i = 0; i < 10; i++) {
    size_t sample_index = rand() % VERIF_SAMPLES;

    classify(&neural_net, (input_data + (NUM_SAMPLES+sample_index)*
          PICTURE_SIZE));

    printf("Expected: %ld\n", getmax(expected_data +
          (NUM_SAMPLES+sample_index)*OUTPUT_LAYER_SIZE,
          OUTPUT_LAYER_SIZE));
  }

  verifyNNet(&neural_net, input_data+NUM_SAMPLES*PICTURE_SIZE,
      expected_data+NUM_SAMPLES*OUTPUT_LAYER_SIZE, VERIF_SAMPLES);

  /*------------------------------------------------------------------------*/
  /*                      Deallocation of resources                         */
  /*------------------------------------------------------------------------*/
  //destroy neural net
  destroyNNet(&neural_net);
  close(expected_data_fd);
  close(input_data_fd);

  free(input_data);
  free(expected_data);

  return 0;
}

/*------------------------------------------------------------------------*/
/*                      Function Definitions                              */
/*------------------------------------------------------------------------*/

void classify(neural_network_t *n_net, float* const input_data) {

  nn_layer_t * last_layer = &n_net->layers_[n_net->num_layers_-1];

  printImage(input_data, PICTURE_SIZE, PIC_WIDTH);
  feedForwardNNet(n_net, input_data);

  printf("Output layer is: \n");
  for (size_t i = 0; i < last_layer->num_neurons_; i++)
    printf("%ld %f\n", i, last_layer->outputs_[i]);
  printf("\n");

  printf("Classified as %ld\n",
      getmax(last_layer->outputs_, last_layer->num_neurons_));
}
