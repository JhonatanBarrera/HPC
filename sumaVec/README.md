## Suma de Vectores.

En este ejercicio se pretende calcular el tiempo de ejecucion de dos vectores con un algoritmo secuencial, el cual luego sera paralelizado.

Para ambos algoritmos se promediara el tiempo de ejecucion con diferentes tamaños de vectores a partir de 20 muestras.  

Datos| _Ts_| _Tp_
-----|-----|------
15000|	0,00003745|	0,0000711
35000|	0,0000872|	0,00010775
55000|	0,00013395|	0,0001513
75000|	0,0002084|	0,00019745
100000|	0,0003041|	0,0002526
150000|	0,0004036|	0,0003631
200000|	0,00057375|	0,00047435
250000|	0,000674275|	0,0005871
300000|	0,00079725|	0,0006729
350000|	0,0008267|	0,00074635
400000|	0,0009476|	0,0008181
450000|	0,00110695|	0,00088275
500000|	0,00118585|	0,0009573
550000|	0,00123735|	0,0010328
600000|	0,00144125|	0,0011103
Tabla 1. Tiempos de ejecucion para datos entre 15000 y 600000.  

![Suma de Vectores](https://github.com/JhonatanBarrera/HPC/blob/master/sumaVec/img/Tiempo_miles1_sumVec.PNG "Tiempo de Ejecucion - Miles")  
Grafica 1. Tiempo de ejecucion para la resolucion de suma de vectores con miles de datos. _CPU_ vs _GPU_.  

En la imagen se observa como para vectores pequeños la _CPU_ toma menor tiempo en resolver el problema de suma de vectores, sin embargo se puede ver que rapidamente la _GPU_ empieza a resolver de manera mas rapida este problema, podemos ver esto en la Grafica 2. Se observa ademas que la ejecucion paralela se matiene de manera mas uniforme que la secuencial.  

Datos | _Ts_ | _Tp_  
------|------|------
5000|	0,0000123|	0,00004105
15000|	0,00003745|	0,0000711
25000|	0,0000614|	0,00008515
35000|	0,0000872|	0,00010775
45000|	0,0001127|	0,00012905
55000|	0,00013395|	0,0001513
65000|	0,0001633|	0,00017585
75000|	0,0002084|	0,00019745
85000|	0,00026175|	0,00021925
95000|	0,0002774|	0,00024085  
Tabla 2. Tiempos de ejecucion para datos entre 5000 y 95000  

![Suma de Vectores](https://github.com/JhonatanBarrera/HPC/blob/master/sumaVec/img/Tiempo_miles2_sumVec.PNG "Tiempo de Ejecucion - Miles")  
Grafica 2. Zoom de tiempo de ejecucion para la resolucion de suma de vectores. _CPU_ vs _GPU_.  

Datos| _Ts_| _Tp_
-----|-----|------
5000000|	0,01175545|	0,00879755
10000000|	0,0234773|	0,0171722
15000000|	0,03544205|	0,02587415
20000000|	0,0471796|	0,0343563
25000000|	0,0589446|	0,0426339
30000000|	0,0706256|	0,0511043
35000000|	0,08237235|	0,0601453
40000000|	0,09430375|	0,0683433
45000000|	0,10592865|	0,0763311
50000000|	0,1181521|	0,0853347
55000000|	0,12966305|	0,09338155
60000000|	0,1414178|	0,1023932
65000000|	0,15392695|	0,11050015  
Tabla 3. Tiempos de ejecucion para datos entre 5000000 y 65000000  

![Suma de Vectores](https://github.com/JhonatanBarrera/HPC/blob/master/sumaVec/img/Tiempo_millones_sumVec.PNG "Tiempo de Ejecucion - Millones")  
Grafica 3. Tiempo de ejecucion para la resolucion de suma de vectores con millones de datos. _CPU_ vs _GPU_.  

La Grafica 3 nos enseña como la resolucion del problema en _GPU_ va mejorando notablemente a medida que el vector es mas grande.  

Tiempo| Aceleracion
------|-------------
300000|	1,184797147
5000000|	1,336218606
10000000|	1,367169029
15000000|	1,369786061
20000000|	1,3732445
25000000|	1,382575838
30000000|	1,381989382
35000000|	1,369555892
40000000|	1,379853621
45000000|	1,387752174
50000000|	1,384572747
55000000|	1,38852964
60000000|	1,381124918
65000000|	1,393002181  
Tabla 4. Aceleracion del algoritmo con el uso de computacion paralela em _GPU_.  

![Suma de Vectores](https://github.com/JhonatanBarrera/HPC/blob/master/sumaVec/img/Aceleracion.PNG "Aceleracion")  
Grafica 4. Aceleracion del algoritmo con el uso de computacion paralela em _GPU_.  

Esta ultima grafica enseña la aceleracion del algoritmo que se presenta con el uso de la computacion paralela en la _GPU_, se puede notar que la aceleracion es mayor con vectores pequeños y que al aumentar este tamaño la aceleracion va siendo constante.
