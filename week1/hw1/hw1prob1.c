/*
 *
 * (c) 2023 Michael Fitzgerald (mpfitz@ucla.edu)
 *
 * This creates a text file containing a parametric curve.  Demonstrates dynamic allocation of
 * arrays and exporting values in a tabular formatted text file.
 *
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

int main(int argc, char* argv[]) {
  static int n_pt_cyc = 100;   // number of points per cycle
  static double tmin = 0.;     // minimum value of t
  static double tmax = 2*M_PI; // maximum value of t

  // create a linearly spaced array of values
  double* t = (double*) malloc(n_pt_cyc * sizeof(double));
  double tstep = (tmax-tmin)/(double)n_pt_cyc; // step size
  if (t == NULL) {
    printf("Failed to allocate memory.\n");
    exit(0);
  }
  for (int i=0; i<n_pt_cyc; i++) {
    t[i] = tmin + (double)i*tstep;
  }

  // compute dependent variables
  double* x = (double*) malloc(n_pt_cyc * sizeof(double));
  double* y = (double*) malloc(n_pt_cyc * sizeof(double));
  for (int i=0; i<n_pt_cyc; i++) {
    x[i] = sin(t[i])*(1-cos(t[i]));
    y[i] = cos(t[i])*(1-cos(t[i])) + 1;
  }
  
  // output values
  static char* fn = "hw1prob1.txt";
  FILE *f = fopen(fn, "w");
  for (int i=0; i<n_pt_cyc; i++) fprintf(f, "%.17g %.17g %.17g\n", t[i], x[i], y[i]);
  fclose(f);
  
  // cleanup
  free(y);
  free(x);
  free(t);
  return 0;
}
