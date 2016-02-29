# HPC
![Imagen HPC de utah.edu](https://www.chpc.utah.edu/_images/banners/hpc1_300.jpg "Data Center")

En el mundo se lleva actualmente un procesamiento masivo de datos tratados para diferentes indoles, y que dependiendo de su complejidad o del numero de operaciones que deban realizar su procesamiento puede tomar un tiempo prolongado, estos problemas pueden reducir sus tiempo considerablemente al ser paralilizados. 

### Programacion Paralela

Cuando se tiene un problema computable, este puede ser dividido en sub-problemas para facilitar la tarea del programador. Esta tecnica sin embargo puede ser usada pensando en otro fin, optimizar el tiempo de ejecucion del problema, esto se logra mediante la compuatzion paralela, en donde muchas instrucciones se ejecutan simultaneamente

La programacion en paralelo muchas veces es simplemente simulada debido al alto costo en hardware, sin embargo en el mercado actual se pueden encontrar tecnologias alternativas. Una de estas tecnologias es la implementacion de la programacion en paralelo que utiliza unidades de procesamiento grafico _GPU_ de _CUDA_.

> CUDA es una arquitectura de cálculo paralelo de NVIDIA que aprovecha la gran potencia de la GPU (unidad de procesamiento gráfico) para proporcionar un incremento extraordinario del rendimiento del sistema. [NVIDIA](http://www.nvidia.es/object/cuda-parallel-computing-es.html)

La ventaja de trabajar con GPU sobre CPU se da en la forma en que se procesan las tareas. Una CPU está formada por varios núcleos optimizados para el procesamiento en serie, las  GPU constan de miles de núcleos más pequeños y eficientes diseñados para manejar múltiples tareas simultáneamente.

![CPU vs GPU](https://github.com/JhonatanBarrera/HPC/blob/master/img/cpu_gpu.png)
