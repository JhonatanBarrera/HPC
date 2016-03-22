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

##### TIEMPO
Matriz|	Secuencial|	Paralelo|	Mem. Com.
-----|-----|-----|-----
96 x 96|	0,00231735|	0,0000816|	0,00006535
192 x 192|	0,01957595|	0,00027805|	0,00015825
288 x 288|	0,0688575|	0,0007673|	0,00037315
384 x 384|	0,18850055|	0,0015989|	0,00067825
480 x 480|	0,33402125|	0,0030189|	0,00116735
576 x 576|	0,61449705|	0,0050489|	0,00181665
672 x 672|	0,92400685|	0,007623|	0,00261075
768 x 768|	1,85664585|	0,011084|	0,0038514
864 x 864|	1,90507285|	0,01580705|	0,00508785
960 x 960|	2,88199695|	0,01995495|	0,00678785
1056 x 1056|	3,7283689|	0,02589315|	0,00824745
1152 x 1152|	5,46513151|	0,03319425|	0,01230315
1248 x 1248|	6,54190065|	0,0419654|	0,01325115
1344 x 1344|	8,5904363|	0,0515892|	0,01676615
1440 x 1440|	10,2623342|	0,06317505|	0,0193159
1536 x 1536|	15,33247625|	0,0758374|	0,0239954
1632 x 1632|	16,4559235|	0,08992915|	0,02757535
1728 x 1728|	22,75213575|	0,10728255|	0,0329679
1824 x 1824|	23,43739485|	0,12486265|	0,03773
1920 x 1920|	33,45206495|	0,1452707|	0,04426265
Tabla 1. Tiempo de ejecución para Matrices de 96x96 hasta 1920x1920 para cada uno de los algoritmos propuestos.

![Multiplicación de Matrices](https://github.com/JhonatanBarrera/HPC/blob/master/multiMat/img/tem_sec.PNG "Tiempo de Ejecución - Secuencial")  
Gráfica 1. Tiempo de ejecución para el algoritmo secuencial.