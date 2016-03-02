#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <nn.h>

#define NUM_LAYERS 3
#define PIC_WIDTH 28
#define PIC_HEIGHT 28
#define PICTURE_SIZE 784 //num pixels 28*28
#define NUM_SAMPLES 50000

#define TRAIN_OFFSET      0x10
#define TRAIN_LAB_OFFSET  0x08

#define OUTPUT_LAYER_SIZE 10

void printImage(double* const data, size_t index);
void classify(neural_network_t *n_net, double* const input_data);
size_t getmax(double* arr, size_t size);

char numToText(double num);

int main(int argc, char ** argv) {
  neural_network_t neural_net;
  size_t layer_sizes[NUM_LAYERS] = {784,30,OUTPUT_LAYER_SIZE};
  //load data

  int input_data_fd = 0;
  int verif_data_fd = 0;

  if (argc > 2) {
    input_data_fd = open(argv[1], O_RDONLY);
    verif_data_fd = open(argv[1], O_RDONLY);
    if ((verif_data_fd == -1) || (input_data_fd == -1)) {
      printf("Please provide input data and verification data\n");
      return 1;
    }
  }
  else {
    printf("Please provide input data and verification data\n");
    return 1;
  }

  //probably not the best idea...
  double* input_data = (double *)
    malloc(NUM_SAMPLES*PICTURE_SIZE*sizeof(double));

  double* verif_data = (double *)
    calloc(NUM_SAMPLES*OUTPUT_LAYER_SIZE,sizeof(double));

  //for NNIST data
  //set input data to first input
  //set output data to first output value
  //should probably mmap the file or something
  lseek(input_data_fd, TRAIN_OFFSET, SEEK_SET);
  lseek(verif_data_fd, TRAIN_LAB_OFFSET, SEEK_SET);

  printf("Copying input data.\n");
  for (size_t i = 0; i < NUM_SAMPLES*PICTURE_SIZE; i++) {
    uint8_t buff = 0;
    read(input_data_fd, &buff, 1);
    input_data[i] = (double) buff;
  }

  printf("Copying verification data and mapping it to vectors.\n");
  for (size_t i = 0; i < NUM_SAMPLES; i++) {
    uint8_t buff = 0;
    read(verif_data_fd, &buff, 1);
    verif_data[(i*OUTPUT_LAYER_SIZE) + (size_t) buff] = 1.0f;
  }

  //TODO fix this
  printf("Checking vectors...\n");
  for (size_t i = 0; i < 20; i++) {
    printf("Vector is: \n");
    for (size_t k = 0; k < last_layer->num_neurons_; k++)
      printf("%ld %f\n", i, last_layer->outputs_[i]);

    printf("Read back as %ld\n",
      getmax(verif_data + i*OUTPUT_LAYER_SIZE,
        OUTPUT_LAYER_SIZE));
  }

  srand(time(NULL));
  initNNet(&neural_net, NUM_LAYERS, layer_sizes);

  //train neural net
  sgdNNet(&neural_net, input_data, verif_data, NUM_SAMPLES, 30, 3.0, 10);

  printf("Verifying Neural Net\n");

  for(int i = 0; i < 10; i++)
    classify(&neural_net, (input_data + i*PICTURE_SIZE));

  //destroy neural net
  destroyNNet(&neural_net);
  close(verif_data_fd);
  close(input_data_fd);
  free(input_data);
  free(verif_data);

  return 0;
}

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
