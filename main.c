#include <stdio.h>
#include <nn.h>

#define NUM_LAYERS 3

int main(int argc, char ** argv) {
  neural_network_t neural_net;
  size_t layer_sizes[NUM_LAYERS] = {784,30,10};
  //load data

  //init neural net
  srand(time(NULL));
  initNNet(&neural_net, NUM_LAYERS, layer_sizes);

  //train neural net

  //destroy neural net
  destroyNNet(&neural_net);

  return 0;
}
