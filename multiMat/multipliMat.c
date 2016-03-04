#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <time.h>

#define row 5
#define col 3

float llenarmat(float *mtrz)
{
	int i, j;
	
	for (i=0;i<row;i++)
		for (j=0;j<col;j++)
			mtrz[i*col+j] = 5.0;
	
	return 0;
}

int main()
{
	//clock_t t_ini, t_fin;
    //double secs;
    
    int lim_v = row * col;
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
    
    int i;
    for (i=0;i<lim_v;i++)
		printf("%f ", mtrz1[i]);
		
	int i;
    for (i=0;i<lim_v;i++)
		printf("%f ", mtrz21[i]);
    
    free(multipli);
    free(mtrz1);
    free(mtrz2);
    
    return 0;
}
