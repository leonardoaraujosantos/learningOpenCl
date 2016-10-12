#include <math.h>
#include <matrix.h>
#include <mex.h>

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  double *a,*b,*result;

  mexPrintf("Hello World\n");
  mexPrintf("%d Outputs on left side\n",nlhs);
  mexPrintf("%d parameters on right side\n",nrhs);

  /* Delcare pointers to inputs and outputs on matlab */
  mxArray *par_mat_1, *par_mat_2, *out_mat_1;

  /* 2 parameters one output*/
  if (nrhs == 2 && nlhs == 1) {
    /*Make a deep copy of the array elements*/
    par_mat_1 = mxDuplicateArray(prhs[0]);
    par_mat_2 = mxDuplicateArray(prhs[1]);

    /* Real data elements in array of type DOUBLE */
    a = mxGetPr(par_mat_1);
    b = mxGetPr(par_mat_2);

    mexPrintf("A=%f B=%f\n",*a,*b);

    /* Create a 1x1 matrix (scalar)*/
    plhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);

    /* Get pointer to output 1x1 matrix (scalar)*/
    result = mxGetPr(plhs[0]);
    *result = *a+*b;
  } else {
    /* Display error message */
    mexErrMsgIdAndTxt( "MATLAB:mexFunction:maxlhs","Too many output arguments.");
  }
}
