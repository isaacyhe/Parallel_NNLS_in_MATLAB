#include "mex.h"
#include "nnls.h"

// MEX Interface for MATLAB
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nrhs != 2) {
        mexErrMsgIdAndTxt("nnls_mex:invalidNumInputs", "Two inputs required (C, d).");
    }
    if (nlhs != 1) {
        mexErrMsgIdAndTxt("nnls_mex:invalidNumOutputs", "One output required (x).");
    }

    // Input Validation
    float *h_C = static_cast<float *>(mxGetData(prhs[0]));
    float *h_d = static_cast<float *>(mxGetData(prhs[1]));
    int m = mxGetM(prhs[0]);
    int n = mxGetN(prhs[0]);

    if (mxGetM(prhs[1]) != m || mxGetN(prhs[1]) != 1) {
        mexErrMsgIdAndTxt("nnls_mex:invalidInput", "Dimensions of d must match the number of rows of C.");
    }

    // Allocate Output
    plhs[0] = mxCreateNumericMatrix(n, 1, mxSINGLE_CLASS, mxREAL);
    float *h_x = static_cast<float *>(mxGetData(plhs[0]));

    // Call NNLS Solver
    NNLS(h_C, h_d, h_x, m, n);
}

