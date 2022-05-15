#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <inttypes.h>

/*		-- TRY CATCH PRINT EXCPEPTION pero CUDA --		*/
#define gpuErrchk(ans) {gpuAssert((ans), __FILE__, __LINE__);}

inline void gpuAssert(cudaError_t code, const char * file, int line, bool abort = true){
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

/*		-- FUNCIONES USADAS POR CPU --		*/

int randInRange(int lower, int upper) {
    return (rand() % (upper - lower + 1)) + lower;
}

void inicializar(short * matrix, int filas, int columnas) {
    for (int i = 0; i < filas; i++) {
        for (int j = 0; j < columnas; j += 4) {

        	//estados
            int random = randInRange(0, 100);
            if (random <= 50) {
                matrix[i * columnas + j] = 4;
                matrix[i * columnas + j + 3] = -1;
            } else {
                random = randInRange(0, 100);
                if (random <= 20) {
                    matrix[i * columnas + j] = 2;
                    matrix[i * columnas + j + 3] = 6;
                } else if (random <= 70) {
                    matrix[i * columnas + j] = 3;
                    matrix[i * columnas + j + 3] = randInRange(0, 6);
                } else {
                    matrix[i * columnas + j] = 1;
                    matrix[i * columnas + j + 3] = randInRange(6, 8);
                }
            }

            //edades
            random = randInRange(0, 100);
            if (random <= 30) {
                matrix[i * columnas + j + 1] = randInRange(0, 104);		//104 semanas = 2 años
            } else if (random <= 84) {
                matrix[i * columnas + j + 1] = randInRange(105, 1976);	//1976 semanas = 38 años
            } else {
                matrix[i * columnas + j + 1] = randInRange(1977, 3640);	//3640 semanas = 70 años
            }

            //heridas
            matrix[i * columnas + j + 2] = randInRange(0, 1);
        }
    }
}

/*		-- FUNCIONES USADAS POR GPU --		*/

__device__ int randomDeviceGenerator(int seed, int lower, int upper) {
	//adaptacion simple de Multiply-with-carry pseudorandom number generator
	//https://en.wikipedia.org/wiki/Multiply-with-carry_pseudorandom_number_generator
    return ((seed * (threadIdx.x + 1) * (blockIdx.x + 1)) % (upper - lower + 1)) + lower;
}

__device__ short * vecindarioDeMoore(short * matrix, int filas, int columnas, int posX, int posY) {

    short * vecindario = (short * ) malloc(3 * 12 * sizeof(short));

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 9; j += 4) {
            int coordX = posX - 4 + j;
            int coordY = posY - 1 + i;
            if ((coordX >= 0) && (coordY >= 0)) {

            	/*
	            caso 1 = ambos positivos
	            caso 2 = alguno es negativo
	            caso 3 = alguno se paso de rango
	            se asigna -1 a todos los atributos de un vecino que no existe
	            */

                if ((coordX < columnas) && (coordY < filas)) {
                    //caso 1
                    vecindario[i * 12 + j] = matrix[coordY * columnas + coordX];
                    vecindario[i * 12 + j + 1] = matrix[coordY * columnas + coordX + 1];
                    vecindario[i * 12 + j + 2] = matrix[coordY * columnas + coordX + 2];
                    vecindario[i * 12 + j + 3] = matrix[coordY * columnas + coordX + 3];
                } else {
                    //caso 3
                    vecindario[i * 12 + j] = -1;
                    vecindario[i * 12 + j + 1] = -1;
                    vecindario[i * 12 + j + 2] = -1;
                    vecindario[i * 12 + j + 3] = -1;
                }
            } else {
                //caso 2
                vecindario[i * 12 + j] = -1;
                vecindario[i * 12 + j + 1] = -1;
                vecindario[i * 12 + j + 2] = -1;
                vecindario[i * 12 + j + 3] = -1;
            }
        }
    }
    return vecindario;
}

__device__ float susceptibilidad(int edad, int heridas) {
    float retorno;

    if (edad < 104) {			//104 semanas = 2 años
        retorno = 0.3;
    } else if (edad < 1976) {	//1976 semanas = 38 años
        retorno = 0.2;
    } else {
        retorno = 0.5;
    }
    if (heridas) {
        retorno += 0.15;
    }
    return retorno;
}

__device__ float porcentajeConSintomas(short * matrix) {
    int cont = 0;

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 12; j += 4) {
            if (((i != 1) && (j != 4)) && (matrix[i * 12 + j] == 2)) {
                cont++;
            }
        }
    }
    return cont / 8;
}

void printMatrix(short * matrix, int filas, int columnas) {
    for (int i = 0; i < filas; i++) {
        for (int j = 0; j < columnas; j += 4) {
            printf(" -- %d %d %d %d", matrix[i * columnas + j], matrix[i * columnas + j + 1], matrix[i * columnas + j + 2], matrix[i * columnas + j + 3]);
        }
        printf("\n");
    }
    printf("\n");
}

__device__ void calcularEstado(int i, int j, short * matrix, short * matrixProxima, int filas, int semilla1, int semilla2, int semilla3, int semilla4){
	//printf("bloque %d thread %d está trabajando la fila %d columna %d\n", blockIdx.x, threadIdx.x, i,j);
	
	j *= 4; 						//una columna esta compuesta de 4 shorts
	int columnas = filas * 4;		//entonces existen (filas * 4) columnas
	int random;

	/*
	cada árbol esta representado por 4 posiciones consecutivas de memoria

	-- posición 0 => estado
	-- posición 1 => edad
	-- posición 2 => heridas (1 si / 0 no)
	-- posición 3 => contador auxiliar multiproposito
	*/

	/* -- INICIO DE: CALCULAR EL PROXIMO ESTADO -- */

    matrixProxima[i * columnas + j + 1] = matrix[i * columnas + j + 1] + 1; //edad++
    switch (matrix[i * columnas + j]) {
        case 0: //podado
            if (matrix[i * columnas + j + 3] == 7) {
                matrixProxima[i * columnas + j] = 4;
                matrixProxima[i * columnas + j + 3] = -1;
            } else {
                matrixProxima[i * columnas + j + 3] = matrix[i * columnas + j + 3] + 1;
            }
        break;

        case 1:
            random = randomDeviceGenerator(semilla1, 0, 100);
            if (matrix[i * columnas + j + 1] <= 104) {			//104 semanas = 2 años
                if (random > 1) {
                    matrixProxima[i * columnas + j] = 4;
                    matrixProxima[i * columnas + j + 3] = -1;
                } else {
                    matrixProxima[i * columnas + j] = 0;
                    matrixProxima[i * columnas + j + 3] = -1;
                }
            } else if (matrix[i * columnas + j + 1] <= 1976) {	//1976 semanas = 38 años
                if (random > 10) {
                    matrixProxima[i * columnas + j] = 4;
                    matrixProxima[i * columnas + j + 3] = -1;
                } else {
                    matrixProxima[i * columnas + j] = 0;
                    matrixProxima[i * columnas + j + 3] = -1;
                }
            } else {
                if (random > 45) {
                    matrixProxima[i * columnas + j] = 4;
                    matrixProxima[i * columnas + j + 3] = -1;
                } else {
                    matrixProxima[i * columnas + j] = 4;
                    matrixProxima[i * columnas + j + 1] = 52;
                    matrixProxima[i * columnas + j + 3] = -1;
                }
            }
        break;

        case 2: //enfermo con sintomas
            if (randomDeviceGenerator(semilla2, 0, 100) <= 90) {
                matrixProxima[i * columnas + j] = 1;
            } else {
                matrixProxima[i * columnas + j] = 2;
            }
            matrixProxima[i * columnas + j + 3] = matrix[i * columnas + j + 3] + 1;
            break;

        case 3: //enfermo sin sintomas
            if (matrix[i * columnas + j + 3] >= 6) {
                matrixProxima[i * columnas + j] = 2;
            } else {
                matrixProxima[i * columnas + j] = 3;
            }
            matrixProxima[i * columnas + j + 3] = matrix[i * columnas + j + 3] + 1;
        break;

        case 4: //sano
            if ((randomDeviceGenerator(semilla3, 0, 100) / 100) <= (porcentajeConSintomas(vecindarioDeMoore(matrix, filas, columnas, j, i)) + susceptibilidad(matrix[i * columnas + j + 1], matrix[i * columnas + j + 2])) * 0.6 + 0.05){
                matrixProxima[i * columnas + j] = 3;
            }
            matrixProxima[i * columnas + j + 3] = -1;
        break;
    }

	//herida aleatoria
    matrixProxima[i * columnas + j + 2] = randomDeviceGenerator(semilla4, 0, 1);

    /* -- FIN DE: CALCULAR EL PROXIMO ESTADO -- */
}

/*		-- KERNEL --		*/

__global__ void simular(
	short * matrix,			//matriz actual
	short * matrixProxima, 	//matriz proxima
	int dimension,			//dimension de la matriz cuadrada
	int cantBloques,		//cantidad de bloques
	int filasPorBloque,		//cantidad de filas que le corresponde calcular a cada bloque
	int moduloBloque,		//cantBloques % filasPorBloque
	int cantThreads,		//cantidad de threads en un bloque
	int columnasPorThread,	//cantidad de columnas que le corresponde calcular a cada thread
	int moduloThread,		//cantThreads % columnasPorThread

	//semillas para eventos random
	int semilla1, int semilla2, int semilla3, int semilla4
	){
    
	int bloqueId = blockIdx.x;
	int threadId = threadIdx.x;

    for(int i = bloqueId * filasPorBloque; i < (bloqueId + 1) * filasPorBloque; i++){
    	for(int j = threadId * columnasPorThread; j < (threadId + 1) * columnasPorThread; j++){
    		calcularEstado(
    			i,					//nro de fila
    			j,					//nro de columna
    			matrix,				//matriz actual
    			matrixProxima,		//matriz proxima
    			dimension,			//dimension de la matriz cuadrada

    			//semillas para eventos random
    			semilla1, semilla2, semilla3, semilla4
			);
    	}

		if(threadId < moduloThread){
			calcularEstado(
				i,											//nro de fila
				columnasPorThread*cantThreads + threadId,	//nro de columna
				matrix,										//matriz actual
				matrixProxima,								//matriz proxima
				dimension,									//dimension de la matriz cuadrada

    			//semillas para eventos random
				semilla1,semilla2, semilla3, semilla4
			);
		}
    }

    if(bloqueId < moduloBloque){
    	for(int j = threadId * columnasPorThread; j < (threadId + 1) * columnasPorThread; j++){
    		calcularEstado(
    			filasPorBloque * cantBloques + bloqueId,	//nro de fila
    			j,											//nro de columna
    			matrix,										//matriz actual
    			matrixProxima,								//matriz proxima
    			dimension,									//dimension de la matriz cuadrada

    			//semillas para eventos random
    			semilla1, semilla2, semilla3, semilla4
			);
    	}
    }

    if(bloqueId < moduloBloque && threadId < moduloThread){
		calcularEstado(
			filasPorBloque * cantBloques + bloqueId,	//nro de fila
			columnasPorThread*cantThreads + threadId,	//nro de columna
			matrix,										//matriz actual
			matrixProxima,								//matriz proxima
			dimension,									//dimension de la matriz cuadrada
	
			//semillas para eventos random
			semilla1, semilla2, semilla3, semilla4
		);
	}
}

/*		-- MAIN --		*/

int main(int argc, char * argv[]) {
    //necesario para generar numeros aleatorios
    srand(time(NULL));

	//argvs
    int filas = atoi(argv[1]);
    int semanas = atoi(argv[2]);
    int cantBloques = atoi(argv[3]);
    int cantThreads = atoi(argv[4]);

	//matriz inicial y su contraparte en gpu
    short * matrix = (short * ) malloc(filas * filas * 4 * sizeof(short));
    short * matrix_gpu;
    
    //matriz proxima y su contraparte en gpu
    short * matrixProxima = (short * ) malloc(filas * filas * 4 * sizeof(short));
    short * matrixProxima_gpu;

	//puntero auxiliar usado para intercambio
    short * auxiliar;

	//calculo de cuanto trabajo va a tener cada bloque
    int moduloBloque = filas % cantBloques;
    int filasPorBloque = floor(filas / cantBloques);

	//calculo de cuanto trabajo va a tener cada thread
    int moduloThread = filas % cantThreads;
    int columnasPorThread = floor(filas / cantThreads);

    //printf("Filas por bloque %d\n\t%d bloque/s tuvieron que calcular otra fila\n\nColumnas por thread %d\n\t%d thread/s tuvieron que calcular otra columna\n\n", filasPorBloque, moduloBloque, columnasPorThread, moduloThread);

    //arranca a medir el tiempo
    struct timespec begin, end;
    clock_gettime(CLOCK_REALTIME, & begin);

	//se inicializa el automata celular (en CPU)
    inicializar(matrix, filas, filas * 4);

	//se ubican las matrices en la memoria de la GPU
    gpuErrchk(cudaMalloc((void ** ) & matrix_gpu, filas * filas * 4 * sizeof(short)));
    gpuErrchk(cudaMalloc((void ** ) & matrixProxima_gpu, filas * filas * 4 * sizeof(short)));

    //CPU -> GPU 
    cudaMemcpy(matrix_gpu, matrix, filas * filas * 4 * sizeof(short), cudaMemcpyHostToDevice);
    cudaMemcpy(matrixProxima_gpu, matrixProxima, filas * filas * 4 * sizeof(short), cudaMemcpyHostToDevice);

    //simulacion!
    for (int generacion = 0; generacion < semanas; generacion++) {

        //ejecutar kernel
        simular <<<cantBloques,cantThreads>>>(
        	matrix_gpu,				//matriz actual
        	matrixProxima_gpu,		//matriz proxima
        	filas,					//dimension de la matriz
        	cantBloques,			//cantidad de bloques
        	filasPorBloque,			//cantidad de filas que le corresponde calcular a cada bloque
        	moduloBloque,			//cantBloques % filasPorBloque
        	cantThreads,			//cantidad de threads
        	columnasPorThread,		//cantidad de columnas que le corresponde calcular a cada thread 
        	moduloThread,			//cantThreads % columnasPorThread

        	//semillas para eventos random
        	randInRange(0, 9), randInRange(0, 9), randInRange(0, 9), randInRange(0, 9)
    	);


        //check si hubo error
        cudaError_t err = cudaGetLastError();
        if (cudaGetLastError() != cudaSuccess) {
            printf("CUDA Error: %s\n", cudaGetErrorString(err));
            exit(-1);
        }

		//intercambio de punteros
        auxiliar = matrix_gpu;
        matrix_gpu = matrixProxima_gpu;
        matrixProxima_gpu = auxiliar;
    }

	//finaliza de medir el tiempo tiempo
    clock_gettime(CLOCK_REALTIME, & end);
    long seconds = end.tv_sec - begin.tv_sec;
    long nanoseconds = end.tv_nsec - begin.tv_nsec;
    double elapsed = seconds + nanoseconds * 1e-9;
    printf("Tiempo medido: %.3f segundos.\n", elapsed);

	/*
	PARA HACERLO COMPARABLE A MPI

	como en mpi no se hace un gather al final
	se deja comentado los cudaMemcpyDeviceToHost
	*/
	
	//GPU -> CPU
	//gpuErrchk(cudaPeekAtLastError());
    //gpuErrchk(cudaMemcpy(matrixProxima, matrixProxima_gpu, filas * filas * 4 * sizeof(short), cudaMemcpyDeviceToHost));

    //gpuErrchk(cudaPeekAtLastError());
    //gpuErrchk(cudaMemcpy(matrix, matrix_gpu, filas * filas * 4 * sizeof(short), cudaMemcpyDeviceToHost));
	
	//print resultado final
    //printMatrix(matrix, filas, filas * 4);

    //frees
    free(matrix);
    free(matrixProxima);
    cudaFree(matrix_gpu);
    cudaFree(matrixProxima_gpu);
    return 0;
}