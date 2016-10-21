/*
Compiling: gcc measureTime.c -pg -O0 -lm -o measureTime
*/
#include <time.h>
#include <stdio.h>      /* printf */
#include <math.h>       /* sqrt */
#include <stdlib.h>
#include <unistd.h> /*usleep*/

float someMatSlowFunc(int iterations)  {
  double sum = 0;
  int i;
  for(i=0; i<iterations; i++){
      sum += log(sqrt((double)i+1)) - log(sqrt((double)i+2));
  }
  /*usleep(1000);*/
  /*sleep(1);*/
  return sum;
}

int main() {
    double result;
    printf("CLOCKS_PER_SEC is %ld\n", CLOCKS_PER_SEC);

    /* Measure elapsed wall time */
    struct timespec now, tmstart;

    clock_gettime(CLOCK_REALTIME, &tmstart);

    result = someMatSlowFunc(4096);

    clock_gettime(CLOCK_REALTIME, &now);

    /* Calculate Wall time in seconds*/
    double seconds = (double)((now.tv_sec+now.tv_nsec*1e-9) - (double)(tmstart.tv_sec+tmstart.tv_nsec*1e-9));
    printf("wall time %fs result=%f\n", seconds, result);

    // measure cpu time
    double start = (double)clock() /(double) CLOCKS_PER_SEC;
    result = someMatSlowFunc(4096);
    double end = (double)clock() / (double) CLOCKS_PER_SEC;
    printf("cpu time %fs result=%f\n", end - start, result);

    return 0;
}
