## HPC
![Imagen HPC de utah.edu](https://www.chpc.utah.edu/_images/banners/hpc1_300.jpg "HPC")

En el mundo se lleva actualmente un procesamiento masivo de datos tratados para diferentes indoles, y que dependiendo de su complejidad o del numero de operaciones que deban realizar su procesamiento puede tomar un tiempo prolongado, estos problemas pueden reducir sus tiempo considerablemente al ser paralelizados.

### Programación Paralela

Cuando se tiene un problema computable, este puede ser dividido en sub-problemas para facilitar la tarea del programador. Esta técnica sin embargo puede ser usada pensando en otro fin, optimizar el tiempo de ejecución del problema, esto se logra mediante la computación paralela, en donde muchas instrucciones se ejecutan simultaneamente

La programación en paralelo muchas veces es simplemente simulada debido al alto costo en hardware, sin embargo en el mercado actual se pueden encontrar tecnologías alternativas. Una de estas tecnologías es la implementación de la programación en paralelo que utiliza unidades de procesamiento gráfico _GPU_ de _CUDA_.

> CUDA es una arquitectura de cálculo paralelo de NVIDIA que aprovecha la gran potencia de la GPU (unidad de procesamiento gráfico) para proporcionar un incremento extraordinario del rendimiento del sistema. [NVIDIA](http://www.nvidia.es/object/cuda-parallel-computing-es.html "CUDA Y EL GPU COMPUTING")  

La ventaja de trabajar con GPU sobre CPU se da en la forma en que se procesan las tareas. Una CPU está formada por varios núcleos optimizados para el procesamiento en serie, las  GPU constan de miles de núcleos más pequeños y eficientes diseñados para manejar múltiples tareas simultáneamente.  

<img src="https://github.com/JhonatanBarrera/HPC/blob/master/path/img/cpu_gpu.png">  

### Métricas de Desempeño

Los algoritmos secuenciales son evaluados de acuerdo a su tiempo de ejecución en función del tamaño del problema. Este tamaño se define con el numero de datos que están involucrados en la ejecución del problema.

En la medición del desempeño de algoritmos paralelos la ejecución no depende solamente del tamaño del problema sino también de la arquitectura y el numero de procesadores. Por esto para evaluar el rendimiento es necesario usar métricas de desempeño como:

* Tiempo de ejecución
* Aceleración  

### Tiempo de ejecución

* _Tiempo de ejecución secuencial:_ Tiempo transcurrido entre el inicio y fin de la ejecución de una tarea.
* _Tiempo de ejecución paralelo:_ Tiempo transcurrido entre el momento en que inicia el computo paralelo y momento en que termina el ultimo procesador.

Se denomina como _Ts_ al tiempo de ejecución secuencial y como _Tp_ al tiempo de ejecución paralelo.

### Aceleración

Al evaluar algoritmos en paralelo nos interesa conocer que tan eficiente es con respecto al algoritmo secuencial. La aceleración es una métrica que nos indica el beneficio relativo de resolver un problema de forma paralela y la denotaremos _S_, esta definida como el cociente del tiempo que tarda en ejecutarse el computo de una tarea mediante el uso de un procesador entre el tiempo que se requiere para realizarlo con _p_ procesadores trabajando en paralelo.

_S = Ts / Tp_

## Hora de Cocinar!

Ejercicios propuestos.

* [Suma de dos vectores.](https://github.com/JhonatanBarrera/HPC/tree/master/sumaVec "sumaVec")
* [Multiplicacion de Matrices.](https://github.com/JhonatanBarrera/HPC/tree/master/multiMat "multiMat")
