/* @file diffusion.c
*  @brief 2D Solver for heat diffusion equation with fixed initial conditions.
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

#define offset 512                          // determines dimension of spatial grid
#define step   0.02                         // spatial discretization step

const int   tfinal = 50;                    // simulation starts at t=0 and ends at tfinal
const float lambda = 0.001 / (step*step);   // lambda from the heat eqaution
const int   dim    = offset + 2;            // dimension of spatial grid
const int   length = offset * offset;       // length of the storage vector

void mv(float* res,float* x)
{
/* @brief Multiply vector x by matrix A of the system of linear equations.
*
*  Multiply vector x by matrix A of the system of linear equations obtained from discretization.
*  Notice that the matrix A does not change during the simulation and therefore we can harcode it
*  into this procedure. This will significantly improve performance.
*
*  @param[in]  x      Pointer to input vector x
*  @param[out] res    Pointer to outpur vector res
*  @return     void
*/
        int tmp = 0;

        for(int i = 0; i < length; i++)
        {
                res[i]= 0;

                tmp = i % offset;

                res[i]+= (1+4*lambda) * x[i];

                res[i]+= (-lambda) * ((i+1 < length && tmp < offset-1 ) ? x[i+1] : 0);

                res[i]+= (-lambda) * ((i+offset < length ) ? x[i+offset] : 0);

                res[i]+= (-lambda) * ((i-1 > -1 && tmp > 0 )? x[i-1] : 0);

                res[i]+= (-lambda) * ((i-offset > -1) ? x[i-offset] : 0);

        }

        return ;
}

float dot(float* a,float* b)
{
/* @brief Calculate the dot product of two vectors.
*
*  @param[in]  a      Pointer to input vector a
*  @param[in]  b      Pointer to input vector b
*  @return     dot    Dot product of a and b
*/
        float res = 0;

        for(int i = 0; i < length; i++)
        {
                res+= a[i] * b[i];
        }

        return res;
}

float norm(float* a)
{
/* @brief Calculate the norm of a vector.
*
*  @param[in]  a      Pointer to input vector a
*  @return     norm   Norm of vector a
*/
        return dot(a,a);
}

void add(float* res,float* a,float* b)
{
/* @brief Calculate the sum of two vectors.
*
*  @param[in]  a      Pointer to input vector a
*  @param[in]  b      Pointer to input vector b
*  @param[out] res    Pointer to output vector res
*  @return     void
*/
        for(int i = 0; i < length; i++)
        {
                res[i] = a[i] + b[i];
        }

        return ;
}

void mul(float* res,float alpha,float* a)
{
/* @brief Multiply a vector by a scalar.
*
*  @param[in]  a      Pointer to input vector a
*  @param[in]  alpha  Scalar alpha
*  @param[out] res    Pointer to output vector res
*  @return     void
*/

        for(int i = 0; i < length; i++)
        {
                res[i] = alpha * a[i];
        }

        return ;
}

void matrixSolver(float* xnew, float* b)
{
/* @brief Solve system of linear equations using conjugate gradients method.
*
*  Solve system of linear equations using conjugate gradients method. The algorithm is iteratively
*  solving the system of equation Ax=b where the matrix A is fixed - it is defined by the equation
*  and its discretization. The matrix is hardcoded in the procedure mv.
*
*  @param[in]  b      Pointer to right hand side b - array of size length
*  @param[out] xnew   Solution of the linear system
*  @return     void
*/
        float epsilon = 0.001;              // precision of the result
        float alpha   = 0;
        float beta    = 0;

        float* xold =  (float*) malloc(sizeof(float)*length);
        float* rold =  (float*) malloc(sizeof(float)*length);
        float* rnew =  (float*) malloc(sizeof(float)*length);
        float* pold =  (float*) malloc(sizeof(float)*length);
        float* pnew =  (float*) malloc(sizeof(float)*length);
        float* tmp  =  (float*) malloc(sizeof(float)*length);
        float* mult =  (float*) malloc(sizeof(float)*length);

        for(int i = 0; i < length; i++)
        {
                xold[i] = 100;
        }

        mv(tmp,xold);

        for(int i = 0; i < length; i++)
        {

                rold[i] = b[i]-tmp[i];
                pold[i] = rold[i];
        }

        while(norm(rold) > epsilon)
        {

                mv(mult,pold);

                alpha = dot(rold,rold)/dot(mult,pold);

                mul(mult,alpha,pold);

                add(xnew,xold,mult);

                mv(mult,pold);

                mul(tmp,-alpha,mult);

                add(rnew,rold,tmp);

                beta = dot(rnew,rnew)/dot(rold,rold);

                mul(mult,beta,pold);

                add(pnew,rnew,mult);

                for(int i = 0; i < length; i++)
                {

                        rold[i] = rnew[i];
                        pold[i] = pnew[i];
                        xold[i] = xnew[i];

                }
        }

        free(xold);
        free(pold);
        free(rold);
        free(pnew);
        free(rnew);
        free(tmp);
        free(mult);

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


