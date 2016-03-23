## Multiplicación de Matrices.

En este ejercicio se analizara el tiempo de ejecución que toma multiplicar dos matrices cuadradas usando un algoritmo secuencial (sobre CPU) contra el tiempo tomado por dos algoritmos en paralelo (sobre GPU).  
Para el primer algoritmo se aprovechara la paralelización sobre GPU sin mayores cambio (uso de cudaMalloc, cudaMemcpy, y llamado al respectivo kernel), para el segundo caso se usara la técnica de tiling buscando, con la que se aprovechara el uso de memoria compartida para obtener una mejor respuesta en el tiempo de ejecución. Se analizara también la aceleración obtenida con cada uno de estos algoritmos paralelos sobre el secuencial. 

Se desarrollara este análisis sobre una maquina con las siguientes características:

* Intel® Core(TM) i7-3770K CPU @ 3.50GHz
* 16 GB de RAM
* NVIDIA® GEFORCE® GTX 780

## Pruebas y Resultados.

Para el análisis de estos algoritmos se tomaran el tiempo de ejecución para DataSet de matrices desde 96x96 hasta 1920x192, el incremento entre cada DataSet es de 96x96.   
Con cada DataSet se obtendrán 20 muestras de tiempo que serán promediadas para tratar de disminuir el ruido que se pueda generar.

Recordar que en el algoritmo de tiling y memoria compartida, se hace uso de tiles que serán de 32x32, por esto mismo se toman matrices cuadradas que sean factores de este valor.



##### Tiempo  

Matriz|	Tiempo CPU|	Tiempo GPU|	Tiempo GPU MC
-----|-----|-----|-----
96 x 96|		0,00231735|		0,0000816|	0,00006535
192 x 192|		0,01957595|		0,00027805|	0,00015825
288 x 288|		0,0688575|		0,0007673|	0,00037315
384 x 384|		0,18850055|		0,0015989|	0,00067825
480 x 480|		0,33402125|		0,0030189|	0,00116735
576 x 576|		0,61449705|		0,0050489|	0,00181665
672 x 672|		0,92400685|		0,007623|	0,00261075
768 x 768|		1,85664585|		0,011084|	0,0038514
864 x 864|		1,90507285|		0,01580705|	0,00508785
960 x 960|		2,88199695|		0,01995495|	0,00678785
1056 x 1056|	3,7283689|		0,02589315|	0,00824745
1152 x 1152|	5,46513151|		0,03319425|	0,01230315
1248 x 1248|	6,54190065|		0,0419654|	0,01325115
1344 x 1344|	8,5904363|		0,0515892|	0,01676615
1440 x 1440|	10,2623342|		0,06317505|	0,0193159
1536 x 1536|	15,33247625|	0,0758374|	0,0239954
1632 x 1632|	16,4559235|		0,08992915|	0,02757535
1728 x 1728|	22,75213575|	0,10728255|	0,0329679
1824 x 1824|	23,43739485|	0,12486265|	0,03773
1920 x 1920|	33,45206495|	0,1452707|	0,04426265
Tabla 1. Tiempo de ejecución (en segundos) para Matrices de 96x96 hasta 1920x1920 para cada uno de los algoritmos propuestos.

![Multiplicación de Matrices - Tiempo Secuencial](https://github.com/JhonatanBarrera/HPC/blob/master/multiMat/img/time_sec_pol_f.PNG "Tiempo de Ejecución - Secuencial")  
Gráfica 1. Tiempo de ejecución para el algoritmo secuencial.  

![Multiplicación de Matrices - Tiempo Paralelo](https://github.com/JhonatanBarrera/HPC/blob/master/multiMat/img/time_par_pol_f.PNG "Tiempo de Ejecución - Paralelo")  
Gráfica 2. Tiempo de ejecución para algoritmos paralelos.  

##### Aceleración  

Datos|	Paralelo (P)|	Tiling + MC (MC)| P vs MC
-----|-----|-----|-----
96|28,39889706|35,46059679|1,248661056
192|70,40442366|123,7026856|1,757030016
288|89,73999739|184,5303497|2,056277636
384|117,8938958|277,9219314|2,357390343
480|110,6433635|286,1363344|2,586113848
576|121,7090951|338,2583602|2,779236507
672|121,2130198|353,9239108|2,919850618
768|167,5068432|482,070377|2,877914525
864|120,5204545|374,4357342|3,106823118
960|144,4251652|424,5817085|2,939804209
1056|143,9905496|452,0632317|3,139534038
1152|164,6409095|444,2058749|2,698028554
1248|155,8879613|493,6855028|3,166925135
1344|166,5161759|512,3678543|3,076985474
1440|162,4428346|531,2894662|3,270624201
1536|202,1756581|638,9756474|3,160497429
1632|182,9876464|596,7620901|3,261215179
1728|212,0767613|690,1299673|3,254151766
1824|187,7054095|621,1872475|3,309373178
1920|230,2739985|755,7628147|3,282015424
Tabla 2. Aceleración obtenida con el uso de los algoritmos paralelos respecto al secuencial, la ultima columna representa la aceleración respecto a los algoritmos paralelos.  

![Multiplicación de Matrices - Aceleración respecto a secuencial](https://github.com/JhonatanBarrera/HPC/blob/master/multiMat/img/aceleration_parsec_pol_f.PNG "Aceleración a partir de algoritmos paralelos")  
Gráfica 3. Aceleración. S vs P (Secuencial vs Paralelo), S vs MC (Secuencial vs Memoria Compartida + Tiling)  

![Multiplicación de Matrices - Aceleración entre paralelos](https://github.com/JhonatanBarrera/HPC/blob/master/multiMat/img/aceleration_par_log_f.PNG "Aceleración entre paralelos")  
Gráfica 4. Aceleración a partir de la optimización del algoritmo con TILING respecto al algoritmo "ingenuo" de paralelización.

## Conclusiones

* Es notorio que los tiempos de ejecución de los algoritmos en paralelo rebozan por mucho la ejecución secuencial. Para todos los DataSet se obtiene mejores tiempos con los algoritmos paralelos desde el inicio de la toma de datos.
* Las lineas de tendencia que mejor se ajustan a los datos para todos los casos son polinomios no necesariamente del mismo orden. Al usar lineas de tendencia polinómicas se debe tener en cuenta los coeficientes de correlación, la varianza y en casos críticos los intervalos de confianza de los datos para establecer cual es el orden del polinomio que mejor se ajusta a los datos.
* El algoritmo de paralelización "ingenuo" para la multiplicación de matrices resulta ser muy eficiente respecto a la ejecución secuencial en CPU, sin embargo obtener datos desde memoria global representa un retraso que puede o no ser innecesario.
* El acceso a memoria compartida y el uso de la técnica de tiling nos permite obtener resultados mas eficientes, esto debido a que estamos aprovechando de mejor manera los recursos de la GPU, optimizando de igual manera el algoritmo de paralelización.
* Se evidencia que al obtener tiempos de ejecución mas pequeños con la técnica de tiling y memoria compartida la aceleración para este algoritmo es mayor.
* La diferencia en la aceleración entre estos algoritmos de paralelización parece que en algún momento tiende a un valor fijo. Se debe recordar que la maquina usada para la toma de los datos no es de uso especifico y esto puede introducir ruido en los tiempos de ejecución.