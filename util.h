#ifndef _UTIL_H_
#define _UTIL_H_

#include <stdlib.h>
#include <stdio.h>

void printImage(float* const data, size_t size, size_t width);

char numToText(float num);

size_t getmax(float* arr, size_t size);
#endif
