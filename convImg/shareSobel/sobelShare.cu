#include <cv.h>
#include <highgui.h>
#include <time.h>
#include <cuda.h>
#include <math.h>

using namespace cv;

#define RED 2
#define GREEN 1
#define BLUE 0

#define MASK_WIDTH 3
#define TILE_SIZE 32

__constant__ char d_M[MASK_WIDTH * MASK_WIDTH];
__constant__ char d_Mt[MASK_WIDTH * MASK_WIDTH];

__global__ void img2gray(unsigned char *imgOutput, unsigned char *imgInput, int width, int height)
{
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;

    if((row < height) && (col < width)){
        imgOutput[row*width+col] = imgInput[(row*width+col)*3+RED]*0.299 + imgInput[(row*width+col)*3+GREEN]*0.587 + imgInput[(row*width+col)*3+BLUE]*0.114;
    }
}

__device__ unsigned char clamp(int value)
{
    if(value < 0)
        value = 0;
    else
        if(value > 255)
            value = 255;
    return (unsigned char)value;
}

__global__ void sobelGradX(unsigned char *imgOutput, int maskWidth, unsigned char *imgInput, int width, int height)
{

		 __shared__ float N_ds[TILE_SIZE + MASK_WIDTH - 1][TILE_SIZE+ MASK_WIDTH - 1]; // Tamanio de la matriz en memoria compartida.
		 
		// Cargar en N_ds elementos de izquierda y derecha.
		int n = maskWidth/2;
    int dest = threadIdx.y*TILE_SIZE+threadIdx.x;
    int destY = dest / (TILE_SIZE+MASK_WIDTH-1);
    int destX = dest % (TILE_SIZE+MASK_WIDTH-1);
    int srcY = blockIdx.y * TILE_SIZE + destY - n; 
    int srcX = blockIdx.x * TILE_SIZE + destX - n;
    int src = (srcY * width + srcX);
    
    // Si srcY y srcX no son elementos fantasmas
    if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width)
        N_ds[destY][destX] = imgInput[src];
    // Si srcY y srcX son elementos fantasma, asigna 0.
    else
        N_ds[destY][destX] = 0;
        
    // Carga en N_ds elementos centrales
    dest = threadIdx.y * TILE_SIZE + threadIdx.x + TILE_SIZE * TILE_SIZE;
    destY = dest /(TILE_SIZE + MASK_WIDTH - 1);
    destX = dest % (TILE_SIZE + MASK_WIDTH - 1);
    srcY = blockIdx.y * TILE_SIZE + destY - n;
    srcX = blockIdx.x * TILE_SIZE + destX - n;
    src = (srcY * width + srcX);
    
    if (destY < TILE_SIZE + MASK_WIDTH - 1) {
        if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width)
            N_ds[destY][destX] = imgInput[src];
        else
            N_ds[destY][destX] = 0;
    }
    
    __syncthreads();

		// Calculo del gradiente.
    int accum = 0;
    int y, x;
    
    for (y = 0; y < maskWidth; y++)
        for (x = 0; x < maskWidth; x++)
            accum += N_ds[threadIdx.y + y][threadIdx.x + x] * d_M[y * maskWidth + x];
            
    y = blockIdx.y * TILE_SIZE + threadIdx.y;
    x = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    if (y < height && x < width)
        imgOutput[(y * width + x)] = clamp(accum);
        
    __syncthreads();
}

__global__ void sobelGradY(unsigned char *imgOutput, int maskWidth, unsigned char *imgInput, int width, int height)
{
    __shared__ float N_ds[TILE_SIZE + MASK_WIDTH - 1][TILE_SIZE+ MASK_WIDTH - 1]; // Tamanio de la matriz en memoria compartida.
		 
		// Cargar en N_ds elementos de izquierda y derecha.
		int n = maskWidth/2;
    int dest = threadIdx.y*TILE_SIZE+threadIdx.x;
    int destY = dest / (TILE_SIZE+MASK_WIDTH-1);
    int destX = dest % (TILE_SIZE+MASK_WIDTH-1);
    int srcY = blockIdx.y * TILE_SIZE + destY - n; 
    int srcX = blockIdx.x * TILE_SIZE + destX - n;
    int src = (srcY * width + srcX);
    
    // Si srcY y srcX no son elementos fantasmas
    if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width)
        N_ds[destY][destX] = imgInput[src];
    // Si srcY y srcX son elementos fantasma, asigna 0.
    else
        N_ds[destY][destX] = 0;
        
    // Carga en N_ds elementos centrales
    dest = threadIdx.y * TILE_SIZE + threadIdx.x + TILE_SIZE * TILE_SIZE;
    destY = dest /(TILE_SIZE + MASK_WIDTH - 1);
    destX = dest % (TILE_SIZE + MASK_WIDTH - 1);
    srcY = blockIdx.y * TILE_SIZE + destY - n;
    srcX = blockIdx.x * TILE_SIZE + destX - n;
    src = (srcY * width + srcX);
    
    if (destY < TILE_SIZE + MASK_WIDTH - 1) {
        if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width)
            N_ds[destY][destX] = imgInput[src];
        else
            N_ds[destY][destX] = 0;
    }
    
    __syncthreads();

		// Calculo del gradiente.
    int accum = 0;
    int y, x;
    
    for (y = 0; y < maskWidth; y++)
        for (x = 0; x < maskWidth; x++)
            accum += N_ds[threadIdx.y + y][threadIdx.x + x] * d_Mt[y * maskWidth + x];
            
    y = blockIdx.y * TILE_SIZE + threadIdx.y;
    x = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    if (y < height && x < width)
        imgOutput[(y * width + x)] = clamp(accum);
        
    __syncthreads();
   
}

__global__ void sobelFilter(unsigned char *imgSobel, unsigned char *sobelOutputX, unsigned char *sobelOutputY, int width, int height)
{
    unsigned int row = blockIdx.y*blockDim.y+threadIdx.y;
    unsigned int col = blockIdx.x*blockDim.x+threadIdx.x;

    if((row < height) && (col < width)){
        imgSobel[row * width + col] = __powf(( __powf(sobelOutputX[row * width + col],2) + __powf(sobelOutputY[row * width + col],2)), 0.5 );
    }
    
}


int main(int argc, char **argv)
{
    clock_t start, end;
    double gpu_time_used;

    char *imageName = argv[1];
    char h_M[] = {-1,0,1,-2,0,2,-1,0,1};
    char h_Mt[] = {-1,-2,-1,0,0,0,1,2,1};
    unsigned char *h_dataRawImage, *d_dataRawImage, *h_imgOutput, *d_imgOutput;
    unsigned char *h_imgSobel, *d_imgSobel, *d_sobelOutputX, *d_sobelOutputY;
    
    Mat image;
    image = imread(imageName, 1);
  
    Size img_size = image.size();

    int width = img_size.width;
    int height = img_size.height;
    int size = sizeof(unsigned char) * width * height * image.channels();
    int sizeGray = sizeof(unsigned char) * width * height;

    h_dataRawImage = (unsigned char*)malloc(size);
    cudaMalloc((void**)&d_dataRawImage,size);

    h_imgOutput = (unsigned char*)malloc(sizeGray);
    cudaMalloc((void**)&d_imgOutput,sizeGray);

    h_imgSobel = (unsigned char*)malloc(sizeGray);
    cudaMalloc((void**)&d_imgSobel,sizeGray);

    cudaMemcpyToSymbol(d_M,h_M,sizeof(char)*MASK_WIDTH*MASK_WIDTH);
    cudaMemcpyToSymbol(d_Mt,h_Mt,sizeof(char)*MASK_WIDTH*MASK_WIDTH);

    cudaMalloc((void**)&d_sobelOutputX,sizeGray);
    cudaMalloc((void**)&d_sobelOutputY,sizeGray);

    h_dataRawImage = image.data;

    start = clock();

    cudaMemcpy(d_dataRawImage, h_dataRawImage, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_M, h_M, sizeof(char)*9, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Mt, h_Mt, sizeof(char)*9, cudaMemcpyHostToDevice);

    int blockSize = 32;
    dim3 dimBlock(blockSize, blockSize, 1);
    dim3 dimGrid(ceil(width/float(blockSize)), ceil(height/float(blockSize)), 1);
    
    img2gray<<<dimGrid,dimBlock>>>(d_imgOutput, d_dataRawImage, width, height);
    cudaDeviceSynchronize();

    // Gradient X
    sobelGradX<<<dimGrid,dimBlock>>>(d_sobelOutputX, 3, d_imgOutput, width, height);
		
	// Gradient Y
    sobelGradY<<<dimGrid,dimBlock>>>(d_sobelOutputY, 3, d_imgOutput, width, height);
	
	// Gradient Magnitude
    sobelFilter<<<dimGrid,dimBlock>>>(d_imgSobel, d_sobelOutputX, d_sobelOutputY, width, height);

    cudaMemcpy(h_imgOutput, d_imgOutput, sizeGray, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_imgSobel, d_imgSobel, sizeGray, cudaMemcpyDeviceToHost);

    end = clock();

    gpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

    
    Mat gray_image, grad_x, abs_grad_x;
    gray_image.create(height,width,CV_8UC1);
    gray_image.data = h_imgOutput;

    Mat sobel_image;
    sobel_image.create(height,width,CV_8UC1);
    sobel_image.data = h_imgSobel;

    namedWindow(imageName, CV_WINDOW_AUTOSIZE);
    namedWindow("Gray Image Secuential", CV_WINDOW_AUTOSIZE);
    namedWindow("Sobel Image OpenCV", CV_WINDOW_AUTOSIZE);

    imshow(imageName,image);
    imshow("Gray Image Secuential", gray_image);
    imshow("Sobel Image OpenCV", sobel_image);
    
    
    printf("%.10f\n",gpu_time_used);

    waitKey(0);
    
    cudaFree(d_dataRawImage);
    cudaFree(d_imgOutput);
    cudaFree(d_M);
    cudaFree(d_Mt);
    cudaFree(d_sobelOutputX);
    cudaFree(d_sobelOutputY);
    cudaFree(d_imgSobel);
    
    // Ocasionan un segmentation default
    //free(h_dataRawImage);
    //free(h_imgOutput);

    return 0;
}
