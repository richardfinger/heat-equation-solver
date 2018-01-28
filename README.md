# heat-equation-solver

 This is a solver for heat diffusion equation using the method of finite differences. The linear system of the method is solved using conjugate gradients algorithm. This solver is implemented twice. In pure C code and in C/CUDA code. The code was implemented as a university assignment to compare the performace of serial CPU code vs parallel GPU code. 

The assignment together with related theory can be found here (written in czech language) :
* [Assignment and theory (CZ)](https://github.com/richardfinger/heat-equation-solver/blob/master/docs/Assignment-theory-cz.pdf)

The solution can be displayed using the make_gif.m Matlab script as :

![alt tag](https://github.com/richardfinger/heat-equation-solver/blob/master/docs/solution.gif)

The source code is lincesed under MIT license.
