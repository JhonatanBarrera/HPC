#include <cv.h>
#include <highgui.h>
#include <time.h>
#include <cuda.h>
#include <math.h>

using namespace cv;

#define RED 2
#define GREEN 1
#define BLUE 0

__global__ void img2gray(unsigned char *imgOutput, unsigned char *imgInput, int width, int height)
{

    for (int row = 0; row < height; row++)
    {
        for (int col = 0; col < width; col++)
        {
            imgOutput[row * width + col] = imgInput[(row * width + col) * 3 + RED] * 0.299 + imgInput[(row * width + col) * 3 + GREEN] * 0.587 + imgInput[(row * width + col) * 3 + BLUE] * 0.114;
        }
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

__global__ void sobelGrad(unsigned char *imgOutput, int maskWidth, char *M, unsigned char *imgInput, int width, int height)
{
    int row;
    int col;
    int Pvalue;

    int N_start_point_row;
    int N_start_point_col;


    for (row = 0; row < height; row++)
    {
        for (col = 0; col < width; col++)
        {
        		Pvalue = 0;
            N_start_point_row = row - (maskWidth/2);
            N_start_point_col = col - (maskWidth/2);
            for(int i = 0; i < maskWidth; i++)
            {
								for(int j = 0; j < maskWidth; j++ )
								{
								    if((N_start_point_col + j >=0 && N_start_point_col + j < width)
								            &&(N_start_point_row + i >=0 && N_start_point_row + i < height))
								    {
								        Pvalue += imgInput[(N_start_point_row + i)*width+(N_start_point_col + j)] * M[i*maskWidth+j];
								        //printf("%d Pvalue: ",Pvalue);
								    }
								}
   					}
   					imgOutput[row * width + col] = clamp(Pvalue);
   					//printf("Pvalue: %d", clamp(Pvalue));
        }
    }
}

__global__ void sobelFilter(unsigned char *imgSobel, unsigned char *sobelOutputX, unsigned char *sobelOutputY, int width, int height)
{
    int row;
    int col;

	for (row = 0; row < height; row++)
    {
        for (col = 0; col < width; col++)
        {
            imgSobel[row * width + col] = (sqrt( pow(sobelOutputX[row * width + col],2) + pow(sobelOutputY[row * width + col],2)) );
        }
    }

}


int main(int argc, char **argv)
{
    double start, end;

    char *imageName = argv[1];
    char h_M[] = {-1,0,1,-2,0,2,-1,0,1}, *d_M;
    char h_Mt[] = {-1,-2,-1,0,0,0,1,2,1}, *d_Mt;
    unsigned char *h_dataRawImage, *d_dataRawImage, *h_imgOutput, *d_imgOutput;
    unsigned char *d_imgSobel, *d_sobelOutputX, *d_sobelOutputY;
    
    Mat image;
    image = imread(imageName, 1);
  
    Size img_size = image.size();

    int width = img_size.width;
    int height = img_size.height;
    int size = sizeof(unsigned char) * width * height * image.channels();
    int sizeGray = sizeof(unsigned char) * width * height;

    dataRawImage = (unsigned char*)malloc(size);
    h_imgOutput = (unsigned char*)malloc(sizeGray);

    cudaMalloc((void**)&d_M,sizeof(char)*9);
    cudaMalloc((void**)&d_Mt,sizeof(char)*9);
    cudaMalloc((void**)&d_sobelOutputX,sizeGray);
    cudaMalloc((void**)&d_sobelOutputY,sizeGray);
    cudaMalloc((void**)&d_imgSobel,sizeGray);
    //sobelOutputX = (unsigned char*)malloc(sizeGray);
    //sobelOutputY = (unsigned char*)malloc(sizeGray);
    //imgSobel = (unsigned char*)malloc(sizeGray);

    dataRawImage = image.data;

    start = clock();

    cudaMemcpy(d_dataRawImage, h_dataRawImage, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_M, h_M, sizeof(char)*9, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Mt, h_Mt, sizeof(char)*9, cudaMemcpyHostToDevice);

    int blockSize = 32;
    dim3 dimBlock(blockSize, blockSize, 1);
    dim3 dimGrid(ceil(width/float(blockSize)), ceil(height/float(blockSize)), 1);
    img2gray<<<dimGrid,dimBlock>>>(d_dataRawImage, width, height, d_imgOutput);
    cudaDeviceSynchronize();
    //img2gray(imgOutput, dataRawImage, width, height);

    Mat gray_image, grad_x, abs_grad_x;
    gray_image.create(height,width,CV_8UC1);
    gray_image.data = imgOutput;

    // Gradient X
    sobelFilter<<<dimGrid,dimBlock>>>(d_imgOutput, width, height, 3, d_M, d_sobelOutputX);
	//sobelGrad(sobelOutputX, 3, M, imgOutput, width, height);
		
	// Gradient Y
    sobelFilter<<<dimGrid,dimBlock>>>(d_imgOutput, width, height, 3, d_M, d_sobelOutputY);
	//sobelGrad(sobelOutputY, 3, Mt, imgOutput, width, height);

	// Gradient Magnitude
    sobelFilter<<<dimGrid,dimBlock>>>(imgSobel, sobelOutputX, sobelOutputY, width, height);
    //sobelFilter(imgSobel, sobelOutputX, sobelOutputY, width, height);

    Mat sobel_image;
    sobel_image.create(height,width,CV_8UC1);
    sobel_image.data = imgSobel;

    namedWindow(imageName, WINDOW_NORMAL);
    namedWindow("Gray Image Secuential", WINDOW_NORMAL);
    namedWindow("Sobel Image OpenCV", WINDOW_NORMAL);

    imshow(imageName,image);
    imshow("Gray Image Secuential", gray_image);
    imshow("Sobel Image OpenCV", sobel_image);

    waitKey(0);

    free(dataRawImage);
    free(imgOutput);
    free(sobelOutputX);
    free(sobelOutputY);
    free(imgSobel);  

    return 0;
}
