#include <util.h>

char numToText(double num) {
  char letter = 0;
  if (num> 229.5/255.0f)
    letter = '@';
  else if (num> 204/255.0f)
    letter = '#';
  else if (num> 178.5/255.0f)
    letter = '8';
  else if (num> 153/255.0f)
    letter = '&';
  else if (num> 127.5/255.0f)
    letter = 'o';
  else if (num> 102/255.0f)
    letter = ';';
  else if (num> 76.5/255.0f)
    letter = '*';
  else if (num> 51/255.0f)
    letter = '.';
  else
    letter = ' ';

  return letter;
}

size_t getmax(mpfr_t* arr, size_t size) {
  size_t max = 0;

  for(size_t i = 0; i < size; i++)
    max = mpfr_cmp(arr[i], arr[max]) > 0 ? i : max;

  return max;
}

size_t getmaxDouble(double* arr, size_t size) {
  size_t max = 0;

  for(size_t i = 0; i < size; i++)
    max = arr[i]> arr[max] ? i : max;

  return max;
}

void printImage(double* const data, size_t size, size_t width) {
  for(size_t i = 0; i < size; i++) {
    printf("%c", numToText(data[i]));
    if(i % width == 0)
      printf("\n");
  }
  printf("\n");
}

