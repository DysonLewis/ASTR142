/*
 *
 * (c) 2023 Michael Fitzgerald (mpfitz@ucla.edu)
 *
 * This performs some number-crunching as a component of a comparison between Python and C.
 *
 *
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

// these are pre-processing variables
#define NX 1024
#define NY 768
#define XMIN -2.5
#define XMAX 1.
#define YMIN -1.
#define YMAX 1.
#define XSTEP (XMAX-XMIN)/(NX-1)
#define YSTEP (YMAX-YMIN)/(NY-1)
#define MAX_ITER 100 // maximum number of iterations
#define R2_MAX 1<<16 // 2e8

double calc_val(double x0, double y0) {
  double ii = 0; // iteration counter; not an int because we modify it in 2nd block below
  double x = 0.;
  double y = 0.;
  double xt, log_zn, nu;
  double log2 = log(2.);

  while ((x*x + y*y <= R2_MAX) && (ii < MAX_ITER)) {
    xt = x*x - y*y + x0;
    y = 2*x*y + y0;
    x = xt;
    ii++;
  }

  if (ii < MAX_ITER) {
    log_zn = log(x*x + y*y) / 2.;
    nu = log(log_zn / log2) / log2;
    ii = ii + 1 - nu;
  }
  
  return ii;
}

int main(int argc, char* argv[]) {
  // set up grid
  double XX[NX];
  double YY[NY];
  double ZZ[NY][NX];
  for (int i=0; i<NX; i++) XX[i] = XMIN+i*XSTEP;
  for (int i=0; i<NY; i++) YY[i] = YMIN+i*YSTEP;

  // run calculation
  for (int i=0; i<NY; i++)
    for (int j=0; j<NX; j++)
      ZZ[i][j] = calc_val(XX[j], YY[i]);

  // output to binary file
  static char* fn = "hw1prob2.dat";
  FILE *f = fopen(fn, "wb");
  fwrite(ZZ, sizeof(double), NX*NY, f);
  fclose(f);

  // cleanup
  return 0;
}
