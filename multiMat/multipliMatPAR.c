#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <malloc.h>
#include <time.h>

#define row 32
#define col 32

__global__ void multipliMat (int *multipli, int *mtrz1, int *mtrz2, int width)
{
  int g_row = blockIdx.y*blockDim.y+threadIdx.y;
  int g_col = blockIdx.x*blockDim.x+threadIdx.x;
  int multi;
  
  if ((g_row<width) && (g_col<width))
  {
    multi = 0;
    for (int k=0; k<width; k++)
    {
      multi += mtrz1[g_row*width+k] * mtrz2[k*width+g_col];
    }
    multipli[g_row*width+g_col] = multi;
  }
}

int llenarmat (int *mtrz)
{
  int i, j;
	
  for (i=0;i<row;i++)
    for (j=0;j<col;j++)
      mtrz[i*col+j] = 7;
  
  return 0;
}

int main()
{
  clock_t t_ini, t_fin;
  double secs;
    
  int *multipli;
  int *mtrz1;
  int *mtrz2;
    
  multipli = NULL;
  mtrz1 = NULL;
  mtrz2 = NULL;
    
  multipli = (int*)malloc(row * col * sizeof(int));
  mtrz1 = (int*)malloc(row * col * sizeof(int));
  mtrz2 = (int*)malloc(row * col * sizeof(int));
    
  llenarmat(mtrz1);
  llenarmat(mtrz2);
  
  int *d_multipli;
  int *d_mtrz1;
  int *d_mtrz2;

  cudaMalloc((void**) &d_multipli, row * col * sizeof(int));
  cudaMalloc((void**) &d_mtrz1, row * col * sizeof(int));
  cudaMalloc((void**) &d_mtrz2, row * col * sizeof(int));
  
  int blockSize = 32;
  dim3 dimBlock(blockSize, blockSize, 1);
  dim3 dimGrid(ceil(col/float(blockSize)), ceil(col/float(blockSize)), 1);
  
  t_ini = clock();
  cudaMemcpy(d_mtrz1, mtrz1, row * col * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_mtrz2, mtrz2, row * col * sizeof(int), cudaMemcpyHostToDevice);
  
  multipliMat<<<dimGrid,dimBlock>>>(d_multipli,d_mtrz1,d_mtrz2, row);
  cudaMemcpy(multipli, d_multipli, row * col  * sizeof(int), cudaMemcpyDeviceToHost);
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
