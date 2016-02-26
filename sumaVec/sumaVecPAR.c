
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <malloc.h>
#include <time.h>

#define n 50000000

// Llenado de vectores con serie 1 + (1+1) + (1+1+1) + .....
int llenarVector(int *vctr){
    int i;
    for (i=0;i<n;i++){
        vctr[i] = i + 1;
    }
    return 0;
}

// Funcion paralelizada
__global__ void sumarVector(int *suma, int *vctr1, int *vctr2){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    suma[i] = vctr1[i] + vctr2[i];
}

// Funcion principal
int main() {
    clock_t t_ini, t_fin;
    double secs;
  int i;
 
 
   
    int *suma;
    int *vctr1;
    int *vctr2;

    suma = NULL;
    vctr1 = NULL;
    vctr2 = NULL;
   
    suma = (int*)malloc(n*sizeof(int));
    vctr1 = (int*)malloc(n*sizeof(int));
    vctr2 = (int*)malloc(n*sizeof(int));
   
    llenarVector(vctr1);
    llenarVector(vctr2);
   
    int *d_suma;
    int *d_vctr1;
    int *d_vctr2;
   
    cudaMalloc((void**) &d_suma, n * sizeof(int));
    cudaMalloc((void**) &d_vctr1, n * sizeof(int));
    cudaMalloc((void**) &d_vctr2, n * sizeof(int));
  
    float blockSize = 32.0;
    float threadSize = ceil(n/blockSize);
  
    for (i=0;i<20;i++)
    {
    t_ini = clock();
    cudaMemcpy(d_vctr1, vctr1, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vctr2, vctr2, n * sizeof(int), cudaMemcpyHostToDevice);
   
    sumarVector<<<blockSize,threadSize>>>(d_suma,d_vctr1,d_vctr2);
    cudaMemcpy(suma, d_suma, n  * sizeof(int), cudaMemcpyDeviceToHost);
    t_fin = clock();
   
    secs = (double)(t_fin - t_ini);
    printf("%f\n", secs / CLOCKS_PER_SEC);
    }
 
    //imprimir(suma);
   
    free(suma);
    free(vctr1);
    free(vctr2);
   
    cudaFree(d_suma);
    cudaFree(d_vctr1);
    cudaFree(d_vctr2);
   
   
    return 0;
}