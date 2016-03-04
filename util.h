#ifndef _UTIL_H_
#define _UTIL_H_

#include <stdlib.h> //for size_t
#include <mpfr.h>   //for mpfr
#include <stdio.h>  //for printf

char numToText(double num);

size_t getmax(mpfr_t* arr, size_t size);
size_t getmaxDouble(double* arr, size_t size);

void printImage(double* const data, size_t size, size_t width);

#endif
