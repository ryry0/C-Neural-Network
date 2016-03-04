#ifndef _UTIL_H_
#define _UTIL_H_

#include <stdlib.h>
#include <stdio.h>

void printImage(long double* const data, size_t size, size_t width);

char numToText(long double num);

size_t getmax(long double* arr, size_t size);
#endif
