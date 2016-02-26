#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <time.h>

#define n 50000000

int llenarVector(int *vctr){
    int i;
    for (i=0;i<n;i++){
        vctr[i] = i + 1;
    }
    return 0;
}

int sumarVector(int *suma, int *vctr1, int *vctr2){
    int i;
    for(i=0;i<n;i++){
        suma[i] = vctr1[i] + vctr2[i];
    }
    return 0;
}

int main(void) {
    clock_t t_ini, t_fin;
    double secs;
    
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
    
  int i;
  for (i=0;i<20;i++){
    t_ini = clock();
    sumarVector(suma,vctr1,vctr2);
    t_fin = clock();
    
  secs = (double)(t_fin - t_ini);
  printf("%f\n", secs / CLOCKS_PER_SEC);
  }
    
    free(suma);
    free(vctr1);
    free(vctr2);
    
    return 0;
}