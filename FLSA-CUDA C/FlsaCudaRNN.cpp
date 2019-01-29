#include "mex.h"
#include "flsarnn.h"

//************************************************
//Main Mex Function
// input argument z,D,y,lambda1,lambda2,alpha
//************************************************
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    float* z = (float*) mxGetPr(prhs[0]); // desired variable z
    float* y = (float*) mxGetPr(prhs[1]); //Second variable D
    float* lambdas = (float*) mxGetPr(prhs[2]); // regularization parameters: lambda1 = lambdas[0]
    
    int signalLength = (int) mxGetM(prhs[1]);//get signal length as the row number of y
    
    plhs[0] = mxCreateNumericMatrix(signalLength-1, 1,mxSINGLE_CLASS, mxREAL);
    float* outFinal = (float*)mxGetData(plhs[0]);
    
    //plhs[1] = mxCreateNumericMatrix(signalLength, 1,mxSINGLE_CLASS, mxREAL);
    //float* out = (float*)mxGetData(plhs[1]);
    
    flsarnn(outFinal, y, z, lambdas,signalLength);
    
    /*plhs[1] = mxCreateDoubleMatrix(signalLength-1, 1, mxREAL);
    float* z_next = (float*)mxGetData(plhs[1]);
    
    zDt(z_next, P_y_DTz, z, lambdas[1], signalLength-1);*/
   
}

