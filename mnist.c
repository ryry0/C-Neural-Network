#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <nn.h>

#define NUM_LAYERS 3
#define INPUT_LAYER_SIZE  784 //have to make sure you can map data to inputs
#define OUTPUT_LAYER_SIZE 10

#define PIC_WIDTH 28
#define PIC_HEIGHT 28
#define PICTURE_SIZE 784 //num pixels 28*28

#define NUM_PICTURES 60000
#define NUM_SAMPLES 50000
#define VERIF_SAMPLES 10000

#define TRAIN_OFFSET      0x10
#define TRAIN_EXP_OFFSET  0x08

#define DEFAULT_TRAIN     "data/train-images-idx3-ubyte"
#define DEFAULT_EXPECT    "data/train-labels-idx1-ubyte"

void printImage(double* const data, size_t index);
void classify(neural_network_t *n_net, double* const input_data);
size_t getmax(double* arr, size_t size);

char numToText(double num);

int main(int argc, char ** argv) {
  neural_network_t neural_net;
  size_t layer_sizes[NUM_LAYERS] = {INPUT_LAYER_SIZE,30,OUTPUT_LAYER_SIZE};
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

  //probably not the best idea...
  double* input_data = (double *)
    malloc(NUM_PICTURES*PICTURE_SIZE*sizeof(double));

  double* expected_data = (double *)
    calloc(NUM_PICTURES*OUTPUT_LAYER_SIZE,sizeof(double));

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
    input_data[i] = (double) buff;
  }

  printf("Copying expected data and mapping it to vectors.\n");
  for (size_t i = 0; i < NUM_PICTURES; i++) {
    uint8_t buff = 0;
    read(expected_data_fd, &buff, 1);
    expected_data[(i*OUTPUT_LAYER_SIZE) + (size_t) buff] = 1.0f;
  }

/*------------------------------------------------------------------------*/
/*           Verifying expected data and representations                  */
/*------------------------------------------------------------------------*/
  printf("Checking verification vectors...\n");
  lseek(expected_data_fd, TRAIN_EXP_OFFSET, SEEK_SET);

  size_t incorrect = 0;
  size_t first_incorrect = 0;
  for (size_t i = 0; i < NUM_PICTURES; i++) {
    uint8_t buff = 0;
    size_t interpreted = 0;
    read(expected_data_fd, &buff, 1);

    //printImage(verif_input_data, i*PICTURE_SIZE);

    //printf("Vector is: \n");
    //for (size_t k = 0; k < OUTPUT_LAYER_SIZE; k++)
      //printf("%ld %f\n", k, verif_expected_data[k + i*OUTPUT_LAYER_SIZE]);

    interpreted = getmax(expected_data + i*OUTPUT_LAYER_SIZE, OUTPUT_LAYER_SIZE);

    if (interpreted != buff) {
      incorrect++;
    }

    //printf("Read back as %ld\n", interpreted);
    //printf("\n");
  }
  printf("Num incorrect %ld\n", incorrect);
  //printf("First incorrect %ld\n", first_incorrect);

/*------------------------------------------------------------------------*/
/*                      Training the neural net                           */
/*------------------------------------------------------------------------*/
  srand(time(NULL));
  initNNet(&neural_net, NUM_LAYERS, layer_sizes);

  //train neural net

  /*
     printf("Printing pre training\n");
     for(size_t i = 1; i < neural_net.num_layers_; i++) {
     for (size_t j = 0; j < neural_net.layers_[i].num_neurons_; j++) {
     printf("layer %ld, errors %ld: %f\n", i, j, neural_net.layers_[i].errors_[j]);
     }
     printf("\n");
     }
     */

  //sgdNNet(n_net, input, expected, #samples in data, epochs, eta, batch)
  /*
  sgdNNet(&neural_net,  //n_net
      input_data,       //input
      expected_data,    //expected
      NUM_SAMPLES,      //#samples in data
      30,               //epochs
      3.0,              //eta
      10,               //batch size
      verif_input_data,             //verification input data
      verif_expected_data,             //verification expected data
      VERIF_SAMPLES);               //verification sample size
      */


  /*
     printf("Printing post training\n");
     for(size_t i = 1; i < neural_net.num_layers_; i++) {
     for (size_t j = 0; j < neural_net.layers_[i].num_neurons_; j++) {
     printf("layer %ld, errors %ld: %f\n", i, j, neural_net.layers_[i].errors_[j]);
     }
     printf("\n");
     }
     */

  /*
     printf("Verifying Neural Net\n");
     for(int i = 0; i < 10; i++)
     classify(&neural_net, (input_data + i*PICTURE_SIZE));
     */

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
void printImage(double* const data, size_t index) {
  for(size_t i = 0; i < PICTURE_SIZE; i++) {
    printf("%c", numToText(data[i + index*PICTURE_SIZE]));
    if(i % PIC_HEIGHT == 0)
      printf("\n");
  }
  printf("\n");
}

char numToText(double num) {
  char letter = 0;
  if (num > 229.5)
    letter = '@';
  else if (num > 204)
    letter = '#';
  else if (num > 178.5)
    letter = '8';
  else if (num > 153)
    letter = '&';
  else if (num > 127.5)
    letter = 'o';
  else if (num > 102)
    letter = ';';
  else if (num > 76.5)
    letter = '*';
  else if (num > 51)
    letter = '.';
  else
    letter = ' ';

  return letter;
}

void classify(neural_network_t *n_net, double* const input_data) {
  nn_layer_t * last_layer = &n_net->layers_[n_net->num_layers_-1];

  printImage(input_data, 0);
  feedForwardNNet(n_net, input_data);

  printf("Output layer is: \n");
  for (size_t i = 0; i < last_layer->num_neurons_; i++)
    printf("%ld %f\n", i, last_layer->outputs_[i]);
  printf("\n");

  printf("Classified as %ld\n",
      getmax(last_layer->outputs_, last_layer->num_neurons_));
}

size_t getmax(double* arr, size_t size) {
  size_t max = 0;

  for(size_t i = 0; i < size; i++)
    max = arr[i] > arr[max] ? i : max;

  return max;
}
