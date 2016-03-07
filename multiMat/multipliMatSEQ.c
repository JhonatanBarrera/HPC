#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <time.h>

#define row 1000
#define col 1000

int llenarmat (int *mtrz)
{
	int i, j;
	
	for (i=0;i<row;i++)
		for (j=0;j<col;j++)
			mtrz[i*col+j] = rand() % 7;
	
	return 0;
}

int multipliMat (int *mtrz1, int *mtrz2, int *multipli)
{
  int i, j, k;
  
  for (i=0;i<row;i++)
  {
    for (j=0;j<col;j++)
    {
			multipli[i*col+j] = 0;
      for (k=0; k<col;k++)
      {
        multipli[i*col+j] += mtrz1[i*col+k] * mtrz2[k*col+j];
      }
    }
  }

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

  t_ini = clock();
  multipliMat(mtrz1, mtrz2, multipli);
  t_fin = clock();
    
  secs = (double)(t_fin - t_ini);
  printf("%f\n", secs / CLOCKS_PER_SEC);
    
  free(multipli);
  free(mtrz1);
  free(mtrz2);
    
  return 0;
}