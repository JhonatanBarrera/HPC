## Detecci�n de Bordes
Los bordes de una imagen digital se pueden definir como transiciones entre dos regiones de niveles de gris significativamente distintos. Suministran una valiosa informaci�n sobre las fronteras de los objetos y puede ser utilizada para segmentar la imagen, reconocer objetos, etc. 
La mayor�a de las t�cnicas para detectar bordes emplean operadores locales basados en distintas aproximaciones discretas de la primera y segunda derivada de los niveles de grises de la imagen. 

##### Operadores basadas en la primera derivada (Gradiente). 
En el caso de funciones bidimensionales f(x,y), la derivada es un vector que apunta en la direcci�n de la m�xima variaci�n de f(x,y) y cuyo m�dulo es proporcional a dicha variaci�n. Este vector se denomina gradiente y se define:

![Gradiente](https://github.com/JhonatanBarrera/HPC/blob/master/convImg/img/gradiente.PNG "Gradiente.")

En el caso bidimensional discreto, las distintas aproximaciones del operador gradiente se basan en diferencias entre los niveles de grises de la imagen. La derivada parcial fx(x,y) ( gradiente de fila GF(i,j) ) puede aproximarse por la diferencia de p�xeles adyacentes de la misma fila. 

![Gradiente X](https://github.com/JhonatanBarrera/HPC/blob/master/convImg/img/gradbix.PNG "Gradiente X.")

La discretizaci�n del vector gradiente en el eje Y (GC(i,j)), ser�: 

![Gradiente Y](https://github.com/JhonatanBarrera/HPC/blob/master/convImg/img/gradbiy.PNG "Gradiente Y.")

El gradiente de la fila GF y de columna GC en cada punto se obtienen mediante la convoluci�n de la imagen con las m�scaras HF y HC, esto es: 

![Convolucion](https://github.com/JhonatanBarrera/HPC/blob/master/convImg/img/gradientes.PNG "Convolucion.")

La magnitud y orientaci�n del vector gradiente suele aproximarse por la expresi�n:

![Magnitud](https://github.com/JhonatanBarrera/HPC/blob/master/convImg/img/magnitud.PNG "Magnitud.")

##### Operador de Sobel
Matem�ticamente, el operador utiliza dos kernels de 3�3 elementos para aplicar convoluci�n a la imagen original para calcular aproximaciones a las derivadas, un kernel para los cambios horizontales y otro para las verticales. Si definimos 'A' como la imagen original, el resultado, que son las dos im�genes 'Gx' y 'Gy' que representan para cada punto las aproximaciones horizontal y vertical de las derivadas de intensidades, es calculado como:

![Sobel](https://github.com/JhonatanBarrera/HPC/blob/master/convImg/img/sobel.PNG "Sobel.")

En cada punto de la imagen, los resultados de las aproximaciones de los gradientes horizontal y vertical pueden ser combinados para obtener la magnitud del gradiente, mediante:

![Magnitud](https://github.com/JhonatanBarrera/HPC/blob/master/convImg/img/magsobel.PNG "Magnitud.")

## Paralelizaci�n del Filtro de Sobel
Para este ejercicion se han construido cuatro versiones del Filtro de Sobel, la primera corresponde a la version secuencial (Ejecutada en CPU) que normalmente se usa en este filtrado, los otros tres algoritmos corresponden a las implementaciones paralelas que han sido mejoradas gradualmente desde el uso de Memoria Global, pasando luego a declarar el kernel de convolucion como una constante en la GPU para final mente hacer uso de la tecnica de Memoria Compartida.

Adicionalmente se construyo una funcion que pasara la imagen original a color a su equivalente en scala de grises.

## Analisis y Resultados
Para el analisis de estos algoritmos se concideraros dos DataSet con imagenes de distintos tamanos, de las cuales se tomaron 20 muestras del tiempo de ejecucion por cada imagen.

##### Tiempo DataSet I

Matriz|	T. CPU|	T. GPU|	T. GPU C| T. GPU MC
-----|-----|-----|-----|-----
1280x800|	0,11442335|	0,00515725|	0,0046692|	0,00464925
1440x900|	0,1435429|	0,00513895|	0,00548835|	0,0055117
1680x1050|	0,18890095|	0,0079691|	0,007093|	0,00709745
1920x1200|	0,24380645|	0,0102462|	0,0091334|	0,00912685
2560x1600|	0,4513609|	0,01778405|	0,0157739|	0,01573345
3840x2400|	0,9742416|	0,039539|	0,03503325|	0,0348836
5120x4096|	2,1714979|	0,08771415|	0,0786137|	0,0782844
6400x4800|	3,2182903|	0,12746205|	0,1146095|	0,1144214
Tabla 1. Tiempo de ejecuci�n (en segundos) para Matrices del DataSet I para cada uno de los algoritmos propuestos.

![Filtro de Sobel - Tiempo Secuencial (DataSet I)](https://github.com/JhonatanBarrera/HPC/blob/master/convImg/img/tiempoSF.PNG "Filtro de Sobel - Tiempo Secuencial (DataSet I)")  
Gr�fica 1.1 Tiempo de ejecuci�n para el algoritmo secuencial (DataSet I).

![Filtro de Sobel - Tiempo Paralelo (DataSet I)](https://github.com/JhonatanBarrera/HPC/blob/master/convImg/img/tiempoPF.PNG "Filtro de Sobel - Tiempo Paralelo (DataSet I)")  
Gr�fica 1.2 Tiempo de ejecuci�n para el algoritmo Paralelo (DataSet I).

##### Tiempo DataSet II

Matriz|	T. CPU|	T. GPU|	T. GPU C| T. GPU MC
-----|-----|-----|-----|-----
580x580|	0,03675095|	0,00191275|	0,00173845|	0,0017324
638x640|	0,0443753|	0,00226315|	0,0020285|	0,0020272
1366x768|	0,1256303|	0,0052891|	0,0047048|	0,0046994
2560x1600|	0,42513105|	0,01774715|	0,0157121|	0,01566325
4928x3264|	1,68560925|	0,0671224|	0,0599379|	0,0599104
5226x4222|	2,29152575|	0,09146755|	0,0824186|	0,0819266
12000x6000|	7,1836216|	0,206442|	0,1775314|	0,26777395
12000x9000|	10,9178012|	0,30800745|	0,26456965|	0,40227365
19843x8504|	16,9437996|	0,4805048|	0,412844|	0,6273025
Tabla 2. Tiempo de ejecuci�n (en segundos) para Matrices del DataSet II para cada uno de los algoritmos propuestos.

![Filtro de Sobel - Tiempo Secuencial (DataSet II)](https://github.com/JhonatanBarrera/HPC/blob/master/convImg/img/tiempoSG.PNG "Filtro de Sobel - Tiempo Secuencial (DataSet II)")  
Gr�fica 2.1. Tiempo de ejecuci�n para el algoritmo secuencial (DataSet II).

![Filtro de Sobel - Tiempo Paralelo (DataSet II)](https://github.com/JhonatanBarrera/HPC/blob/master/convImg/img/tiempoPG.PNG "Filtro de Sobel - Tiempo Paralelo (DataSet II)")  
Gr�fica 2.2. Tiempo de ejecuci�n para el algoritmo Paralelo (DataSet II).

##### Aceleraci�n DataSet I

Matriz|	S vs G|	S vs C|	S vs MC| C vs MC
-----|-----|-----|-----|-----
1280x800|	22,18689224|	24,50598604|	24,61114158|	1,004291015
1440x900|	27,93234026|	26,15410825|	26,04330787|	0,995763558
1680x1050|	23,70417613|	26,63202453|	26,61532663|	0,999373014
1920x1200|	23,79481661|	26,69394202|	26,71309926|	1,000717663
2560x1600|	25,38009621|	28,61441368|	28,68798007|	1,002570956
3840x2400|	24,64001619|	27,80905568|	27,92835602|	1,004289982
5120x4096|	24,75652902|	27,62238516|	27,73857755|	1,004206457
6400x4800|	25,24900784|	28,0804846|	28,12664676|	1,001643923
Tabla 3. Aceleraci�n obtenida con el uso de los algoritmos paralelos respecto al secuencial en el DataSet I, la ultima columna representa la aceleraci�n respecto a los algoritmos paralelos con uso de Kernel Constante y Memoria Compartida. 

![Filtro de Sobel - Aceleraci�n respecto a secuencial (DataSet I)](https://github.com/JhonatanBarrera/HPC/blob/master/convImg/img/aceleracionPF.PNG "Aceleraci�n a partir de algoritmos paralelos (DataSet I)")  
Gr�fica 3.1 Aceleraci�n. S vs G (Secuencial vs Global), S vs C (Secuencial vs Kernel Constante), S vs MC (Secuencial vs Memoria Compartida).  

![Filtro de Sobel - Aceleraci�n entre paralelos (DataSet I)](https://github.com/JhonatanBarrera/HPC/blob/master/convImg/img/aceleracionCMCF.PNG "Aceleraci�n entre paralelos (DataSet I)")  
Gr�fica 3.2. Aceleraci�n a partir de la optimizaci�n del algoritmo con Kernel Constante, donde se hizo uso de Memoria Compartida.

##### Aceleraci�n DataSet II

Matriz|	S vs G|	S vs C|	S vs MC| C vs MC
-----|-----|-----|-----|-----
580x580|	19,21367142|	21,1400673|	21,21389402|	1,003492265
638x640|	19,6077591|	21,87591817|	21,88994672|	1,000641279
1366x768|	23,75268004|	26,70258034|	26,73326382|	1,001149083
2560x1600|	23,95489135|	27,05755755|	27,14194372|	1,003118765
5226x4222|	25,11246991|	28,12259439|	28,13550318|	1,000459019
4928x3264|	25,05288214|	27,80350249|	27,97047296|	1,006005376
12000x9000|	34,79728737|	40,46394948|	26,82718614|	0,662989809
12000x6000|	35,4465491|	41,26626467|	27,14023451|	0,657685757
19843x8504|	35,26249811|	41,04165157|	27,01057241|	0,658125864
Tabla 4. Aceleraci�n obtenida con el uso de los algoritmos paralelos respecto al secuencial en el DataSet II, la ultima columna representa la aceleraci�n respecto a los algoritmos paralelos con uso de Kernel Constante y Memoria Compartida.

![Filtro de Sobel - Aceleraci�n respecto a secuencial (DataSet II)](https://github.com/JhonatanBarrera/HPC/blob/master/convImg/img/aceleracionPG.PNG "Aceleraci�n a partir de algoritmos paralelos (DataSet II)")  
Gr�fica 4.1 Aceleraci�n. S vs G (Secuencial vs Global), S vs C (Secuencial vs Kernel Constante), S vs MC (Secuencial vs Memoria Compartida).

![Filtro de Sobel - Aceleraci�n entre paralelos (DataSet II)](https://github.com/JhonatanBarrera/HPC/blob/master/convImg/img/aceleracionCMCG.PNG "Aceleraci�n entre paralelos (DataSet II)")  
Gr�fica 4.2. Aceleraci�n a partir de la optimizaci�n del algoritmo con Kernel Constante, donde se hizo uso de Memoria Compartida.

## Conclusiones

* Para ambos DataSet los tiempos de ejecuci�n de los algoritmos en paralelo rebozan por mucho la ejecuci�n del secuencial al ser estos primeros menores. Se obtiene entonces mejores tiempos con las implementaciones en paralelos.
* Basado en las experiencias anteriores se esperar�a que la implementaci�n que hace uso de la memoria compartida sea la que siempre de menores tiempos de ejecuci�n y una mayor aceleraci�n, lo cual es valido para una cantidad de datos, sin embargo encontramos un punto en el que los tiempos de esta implementaci�n se disparan y la aceleraci�n disminuye.
* El algoritmo de paralelizaci�n con Kernel Constante para el Filtro de Sobel resulta ser muy eficiente respecto a las otras implementaciones en paralelo.
* El acceso a memoria compartida y el uso de la t�cnica de tiling nos permite ahorrar transacciones de memoria global, esto debido a que estamos aprovechando de mejor manera los recursos de la GPU, sin embargo no en todos los casos se logra optimizar el algoritmo con esta t�cnica.
* Se evidencia que no todas las t�cnicas que se pueden llegar a usar en la paralelizaci�n son eficientes para todas las implementaciones.
* En algoritmos de filtrado de im�genes, los tiempos pueden variar dependiendo de la imagen de entrada, independiente de si es del mismo tama�o.
* La diferencia en la aceleraci�n entre estos algoritmos de paralelizaci�n parece que en alg�n momento tiende a un valor fijo. Se debe recordar que la maquina usada para la toma de los datos no es de uso especifico y esto puede introducir ruido en los tiempos de ejecuci�n.