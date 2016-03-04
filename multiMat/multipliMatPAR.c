#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <malloc.h>
#include <time.h>

#define row 1000
#define col 1000

float llenarmat (float *mtrz)
{
	int i, j;
	
	for (i=0;i<row;i++)
		for (j=0;j<col;j++)
			mtrz[i*col+j] = rand() % 7;
	
	return 0;
}

_global_ void multipliMat (float *mtrz1, float *mtrz2, float *multipli)
{
  
}

int main()
{
  clock_t t_ini, t_fin;
  double secs;
    
  float *multipli;
  float *mtrz1;
  float *mtrz2;
    
  multipli = NULL;
  mtrz1 = NULL;
  mtrz2 = NULL;
    
  multipli = (float*)malloc(row * col * sizeof(float));
  mtrz1 = (float*)malloc(row * col * sizeof(float));
  mtrz2 = (float*)malloc(row * col * sizeof(float));
    
  llenarmat(mtrz1);
  llenarmat(mtrz2);
  
  float *d_multipli;
  float *d_mtrz1;
  float *d_mtrz2;

  cudaMalloc((void**) &d_multipli, row * col * sizeof(float));
  cudaMalloc((void**) &d_mtrz1, row * col * sizeof(float));
  cudaMalloc((void**) &d_mtrz2, row * col * sizeof(float));
  
  float blockSize = 1024.0;
  float dimGrid = ceil((row * col)/blockSize);
  
  t_ini = clock();
  cudaMemcpy(d_mtrz1, mtrz1, row * col * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_vctr2, mtrz2, row * col * sizeof(float), cudaMemcpyHostToDevice);
  
  sumarVector<<<dimGrid,blockSize>>>(d_multipli,mtrz1,mtrz2);
  cudaMemcpy(multipli, d_multipli, row * col  * sizeof(float), cudaMemcpyDeviceToHost);
  t_fin = clock();
    
  secs = (double)(t_fin - t_ini);
  printf("%f\n", secs / CLOCKS_PER_SEC);
    
  free(multipli);
  free(mtrz1);
  free(mtrz2);
  
  cudaFree(d_multipli);
  cudaFree(d_mtrz1);
  cudaFree(d_mtrz2);
    
  return 0;
}