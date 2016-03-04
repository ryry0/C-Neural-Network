#ifndef _UTIL_H_
#define _UTIL_H_

size_t getmax(mpfr_t* arr, size_t size) {
  size_t max = 0;

  for(size_t i = 0; i < size; i++)
    max = mpfr_cmp(arr[i], arr[max]) > 0 ? i : max;

  return max;
}

void printImage(mpfr_t* const data, size_t size, size_t width) {
  for(size_t i = 0; i < size; i++) {
    printf("%c", numToText(data[i]));
    if(i % width == 0)
      printf("\n");
  }
  printf("\n");
}

char numToText(mpfr_t num) {
  char letter = 0;
  if (mpfr_cmp_d(num, 229.5/255.0f) > 0)
    letter = '@';
  else if (mpfr_cmp_d(num > 204/255.0f) > 0)
    letter = '#';
  else if (mpfr_cmp_d(num > 178.5/255.0f) > 0)
    letter = '8';
  else if (mpfr_cmp_d(num > 153/255.0f) > 0)
    letter = '&';
  else if (mpfr_cmp_d(num > 127.5/255.0f) > 0)
    letter = 'o';
  else if (mpfr_cmp_d(num > 102/255.0f) > 0)
    letter = ';';
  else if (mpfr_cmp_d(num > 76.5/255.0f) > 0)
    letter = '*';
  else if (mpfr_cmp_d(num > 51/255.0f) > 0)
    letter = '.';
  else
    letter = ' ';

  return letter;
}

#endif
