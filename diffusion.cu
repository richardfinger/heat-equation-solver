/* @file diffusion.cu
*  @brief 2D Solver for heat diffusion equation with fixed initial conditions written to run on GPU.
*
*  This is a solver for diffusion equation using the method of finite differences. The linear system
*  of the method is solved using conjugate gradients. It solves the equation du/dt - lambda*L^2u = 0
*  where L is the Laplacian operator, u is temperature and t is time. It solves this equation in
*  time and space. The time domain is defined by tfinal, that is [0,tfinal]. The spatial domain is
*  defined by step and offset, where step is the distance of the gridpoints in meters and offset is
*  the number of grid points in both directions ( the grid is a square ). The initial condition is
*  hardcoded into the procedure initializeMatrix. The boundary condition is harcoded in the
*  procedure run. All of the input of this program is hardcoded - the intention was to demonstrate
*  the performance difference between C and CUDA code, not to make a general solver. The output of
*  this program is a set of files named matrix<t>.csv containing the 2D temperature function at time
*  t.

*  @author Richard Finger
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>

#define BLOCK_SIZE 512                      // CUDA block size
#define offset 512                          // determines dimension of spatial grid
#define step   0.02                         // spatial discretization step

const int   tfinal = 50;                    // simulation starts at t=0 and ends at tfinal
const float lambda = 0.001 / (step*step);   // lambda from the heat eqaution
const int   dim    = offset + 2;            // dimension of spatial grid
const int   length = offset * offset;       // length of the storage vector

__global__ void cudamv(float* res,float* y)
{
/* @brief Multiply vector x by matrix A of the system of linear equations on GPU.
*
*  Multiply vector x by matrix A of the system of linear equations obtained from discretization.
*  Notice that the matrix A does not change during the simulation and therefore we can harcode it
*  into this procedure. This will significantly improve performance.
*
*  @param[in]  y      Pointer to input vector y
*  @param[out] res    Pointer to outpur vector res
*  @return     void
*/

        __shared__ float  x[BLOCK_SIZE + 2*offset];

        int g_i = threadIdx.x + blockIdx.x * blockDim.x;
        int i   = threadIdx.x + offset;

        x[i] = y[g_i];

        if(threadIdx.x < offset)
        {
                x[threadIdx.x ] = (g_i - offset) > 0 ? y[(g_i - offset)] : 0;
                x[threadIdx.x + BLOCK_SIZE + offset] = (g_i + BLOCK_SIZE) < length ? y[g_i + BLOCK_SIZE] : 0;
        }

        __syncthreads();

        float result = 0;

        int tmp = g_i % offset;

        result += (1 + 4*lambda) * x[i];

        result += (-lambda) * ((g_i+1 < length && tmp < offset -1) ? x[i+1] :0);

        result += (-lambda) * ((g_i+offset < length ) ? x[i+offset] :0);

        result += (-lambda) * ((g_i-1 > -1 && tmp > 0 ) ? x[i-1] :0);

        result += (-lambda) * ((g_i-offset > -1) ? x[i-offset] :0);

        res[g_i] = result;
}

__global__ void cudadot(float* res,float* a, float* b)
{
/* @brief Calculate the dot product of two vectors on GPU.
*
*  @param[in]  a      Pointer to input vector a
*  @param[in]  b      Pointer to input vector b
*  @param[out] res    Dot product of a and b
*  @return     void
*/
        unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

        for(int i = 0; i < 2; i++)
        {
                res[index + i*length/2] = a[index + i*length/2] * b[index + i*length/2];
        }

        __shared__ volatile float sdata[BLOCK_SIZE];

        unsigned int t_id = threadIdx.x;

        sdata[t_id] = res[index] + res[index + length/2];

        __syncthreads();

        if(BLOCK_SIZE >= 256)
        {
                if(t_id < 128)
                {
                        sdata[t_id] += sdata[t_id + 128];
                }
                __syncthreads();
        }
        if(BLOCK_SIZE >= 128)
        {
                if(t_id < 64)
                {
                        sdata[t_id] += sdata[t_id + 64];
                }
                __syncthreads();
        }
        if(t_id < 32)
        {
                sdata[t_id] += sdata[t_id + 32];
                sdata[t_id] += sdata[t_id + 16];
                sdata[t_id] += sdata[t_id + 8];
                sdata[t_id] += sdata[t_id + 4];
                sdata[t_id] += sdata[t_id + 2];
                sdata[t_id] += sdata[t_id + 1];
        }

        if(t_id == 0) res[blockIdx.x] = sdata[0];
}

float dot(float* d_res,float* d_a,float* d_b)
{
/* @brief Calculate the dot product of two vectors.
*
*  @param[in]  d_a    Pointer to input vector d_a
*  @param[in]  d_b    Pointer to input vector d_b
*  @param[in]  d_res  Pointer to intermediate result vector d_res
*  @return     dot    Dot product of d_a and d_b
*/

        cudadot<<<length/BLOCK_SIZE,BLOCK_SIZE/2>>>(d_res,d_a,d_b);

        float* res = (float*) malloc(sizeof(float)*length/BLOCK_SIZE);

        cudaMemcpy(res,d_res,sizeof(float)*length/BLOCK_SIZE,cudaMemcpyDeviceToHost);

        float result = 0;

        for(int i = 0; i < length/BLOCK_SIZE; i++)
        {
                result += res[i];
        }

        free(res);

        return result;
}

float norm(float* res, float* a)
{
/* @brief Calculate the norm of a vector.
*
*  @param[in]  a      Pointer to input vector a
*  @param[in]  res    Pointer to intermediate result vector res
*  @return     norm   Norm of vector a
*/
        return dot(res,a,a);
}

__global__ void cudaadd(float* res, float* a, float* b)
{
/* @brief Calculate the sum of two vectors on GPU.
*
*  @param[in]  a      Pointer to input vector a
*  @param[in]  b      Pointer to input vector b
*  @param[out] res    Pointer to output vector res
*  @return     void
*/
        unsigned int g_id = blockIdx.x*blockDim.x + threadIdx.x;

        res[g_id] = a[g_id] + b[g_id];
}

__global__ void cudasub(float* res, float* a, float* b)
{
/* @brief Calculate the difference of two vectors on GPU.
*
*  @param[in]  a      Pointer to input vector a
*  @param[in]  b      Pointer to input vector b
*  @param[out] res    Pointer to output vector res
*  @return     void
*/
        unsigned int g_id = blockIdx.x*blockDim.x + threadIdx.x;

        res[g_id] = a[g_id] - b[g_id];
}


__global__ void cudamul(float* res, float alpha, float* b)
{
/* @brief Multiply a vector by a scalar on GPU.
*
*  @param[in]  b      Pointer to input vector b
*  @param[in]  alpha  Scalar alpha
*  @param[out] res    Pointer to output vector res
*  @return     void
*/
        unsigned int g_id = blockIdx.x*blockDim.x + threadIdx.x;

        res[g_id] = alpha * b[g_id];
}

void matrixSolver(float* host_xnew,float* b)
{
/* @brief Solve system of linear equations using conjugate gradients method.
*
*  Solve system of linear equations using conjugate gradients method. The algorithm is iteratively
*  solving the system of equation Ax=b where the matrix A is fixed - it is defined by the equation
*  and its discretization. The matrix is hardcoded in the procedure mv.
*
*  @param[in]  b           Pointer to right hand side b - array of size length
*  @param[out] host_xnew   Solution of the linear system
*  @return     void
*/
        float epsilon = 0.001;              //Precision of the solution

        int size = length * sizeof(float);

        float* xold ;
        float* xnew ;
        float* rold ;
        float* rnew ;
        float* pold ;
        float* pnew ;
        float* tmp  ;
        float* mult ;
        float* res ;
        float* d_b;


        cudaMalloc((void**) &d_b , size);
        cudaMalloc((void**) &xold, size);
        cudaMalloc((void**) &xnew, size);
        cudaMalloc((void**) &rold, size);
        cudaMalloc((void**) &rnew, size);
        cudaMalloc((void**) &pold, size);
        cudaMalloc((void**) &pnew, size);
        cudaMalloc((void**) &tmp , size);
        cudaMalloc((void**) &mult, size);
        cudaMalloc((void**) &res , size);

        cudaMemset(d_b , 0, size);
        cudaMemset(xold, 0, size);
        cudaMemset(xnew, 0, size);
        cudaMemset(rold, 0, size);
        cudaMemset(rnew, 0, size);
        cudaMemset(pold, 0, size);
        cudaMemset(pnew, 0, size);
        cudaMemset(tmp , 0, size);
        cudaMemset(mult, 0, size);
        cudaMemset(res , 0, size);

        cudaMemcpy(d_b,b,size,cudaMemcpyHostToDevice);

        cudamv<<<length/BLOCK_SIZE,BLOCK_SIZE>>>(tmp,xold);

        cudasub<<<length/BLOCK_SIZE,BLOCK_SIZE>>>(rold,d_b,tmp);

        cudaMemcpy(pold,rold,size,cudaMemcpyDeviceToDevice);

        float alpha = 0;
        float beta  = 0;

        while(norm(res,rold) > epsilon)
        {
                cudamv<<<length/BLOCK_SIZE,BLOCK_SIZE>>>(mult,pold);

                alpha = dot(res,rold,rold)/dot(res,mult,pold);

                cudamul<<<length/BLOCK_SIZE,BLOCK_SIZE>>>(mult,alpha,pold);

                cudaadd<<<length/BLOCK_SIZE,BLOCK_SIZE>>>(xnew,xold,mult);

                cudamv<<<length/BLOCK_SIZE,BLOCK_SIZE>>>(mult,pold);

                cudamul<<<length/BLOCK_SIZE,BLOCK_SIZE>>>(tmp,-alpha,mult);

                cudaadd<<<length/BLOCK_SIZE,BLOCK_SIZE>>>(rnew,rold,tmp);

                beta = dot(res,rnew,rnew)/dot(res,rold,rold);

                cudamul<<<length/BLOCK_SIZE,BLOCK_SIZE>>>(mult,beta,pold);

                cudaadd<<<length/BLOCK_SIZE,BLOCK_SIZE>>>(pnew,rnew,mult);

                cudaMemcpy(rold,rnew,size,cudaMemcpyDeviceToDevice);
                cudaMemcpy(pold,pnew,size,cudaMemcpyDeviceToDevice);
                cudaMemcpy(xold,xnew,size,cudaMemcpyDeviceToDevice);
        }

        cudaMemcpy(host_xnew,xnew,size,cudaMemcpyDeviceToHost);

        cudaFree(xold);
        cudaFree(xnew);
        cudaFree(pold);
        cudaFree(rold);
        cudaFree(pnew);
        cudaFree(rnew);
        cudaFree(tmp);
        cudaFree(mult);
        cudaFree(res);
        cudaFree(d_b);

        return ;
}

void printMatrix(int iFile, float uold[][dim])
{
/* @brief Print matrix into a file called matrix<iFile>.csv .
*
*  @param[in]  iFile  File number
*  @param[in]  uold   Matrix to print
*  @return     void
*/
        char str[14];
        char num[4];

        strcpy(str,"matrix");

        num[0] = '0' + (iFile % 10);
        num[1] = '\0';

        if(iFile > 9)
        {
                num[0] = '0' + (iFile-(iFile % 10))/10;
                num[1] = '0' + (iFile % 10);
                num[2] = '\0';

        }

        if(iFile > 99)
        {
                num[0] = '0' + (iFile-(iFile % 100))/100;
                num[1] = '0' + ((iFile-(iFile % 10))/10 % 10);
                num[2] = '0' + (iFile % 10);
                num[3] = '\0';
        }

        strcat(str,num);

        strcat(str,".csv");

        FILE* f = fopen(str,"w");

        for(int i = 0; i < dim; i++)
        {
                for(int j = 0; j < dim; j++)
                {
                        fprintf(f, "%f, ",uold[i][j]);
                }

                fprintf(f, "\n");
        }

        fclose(f);
}

void initializeMatrix(float uold[][dim])
{
/* @brief Initializes the solution matrix - the initial conditions are hard coded.
*
*  This procedure initializes the matrix uold to -1 everywhere except in two elipses centered at
*  [10,10] and [25,30] where the value is 10.
*
*  @param[out] uold   Solution matrix
*  @return     void
*/
        for(int i = 0; i < dim; i++)
        {
                for(int j = 0; j < dim; j++)
                {
                        if( ((i - 10)*(i - 10) < 10 && (j - 10)*(j - 10) < 17 ) ||
                            ((i - 25)*(i - 25) < 17 && (j - 30)*(j - 30) < 10 )  )
                        {
                                uold[i][j] = 10;
                        }
                        else
                        {
                                uold[i][j] = -1;
                        }
                }
        }
}

void run()
{
/* @brief Runs the algorithm to solve 2D heat equation and prints the result into files.
*
*  This procedure drives the solver. It initializes the temperature matrix uold and solves the
*  linear system of equations to obtain the new temperature unew. It sets the boundary condition and
*  prints the results into files named matrix<t>.csv where t is time. For the purposes of the
*  calculation the 2D matrix uold is stored into 1D vector x.

*  @param[out] uold   Solution matrix
*  @return     void
*/

        float uold[dim][dim];
        float unew[dim][dim];
        float b[length];
        float x[length];
        int t     = 0;
        int iFile = 1;

        initializeMatrix(uold);

        while(t <= tfinal)
        {
                printf("Calculating t = %d\n",t);

                // Initialize the right hand side b
                for(int k = 0; k < length; k++)
                {
                        int j2 = k % (dim-2);
                        int i2 = (k-j2) / (dim-2);

                        b[k] = uold[i2+1][j2+1];

                        if(i2 == 0 || i2 == dim-3) b[k]+= - lambda;
                        if(j2 == 0 || j2 == dim-3) b[k]+= - lambda;
                }

                matrixSolver(x,b);

                // Translate 1D vector into 2D matrix
                for(int k = 0; k < length; k++)
                {
                        int j2 = k % (dim-2);
                        int i2 = (k-j2)/(dim-2);

                        unew[i2+1][j2+1] = x[k];
                }

                // Set boundary condition and update solution
                for(int i = 0; i < dim; i++)
                {
                        for(int j = 0; j < dim; j++)
                        {
                                if(i == 0 || i== dim-1 || j == 0 || j == dim-1)
                                {
                                        uold[i][j] = -1;
                                }
                                else
                                {
                                        uold[i][j] = unew[i][j];
                                }
                        }
                }

                t++;

                if(0 == t % 1 )
                {
                        printMatrix(iFile,uold);

                        iFile++;
                }

        }
}

int main(int argc,char** argv)
{
/* @brief Main.
*
*  @param[in]  argc
*  @param[in]  argv
*  @return     int
*/

        run();

        return 0;
}


