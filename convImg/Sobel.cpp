#include <cv.h>
#include <highgui.h>

using namespace cv;

#define RED 2
#define GREEN 1
#define BLUE 0

void img2gray(unsigned char *imgOutput, unsigned char *imgInput, int width, int height)
{

    for (int row = 0; row < height; row++)
    {
        for (int col = 0; col < width; col++)
        {
            imgOutput[row * width + col] = imgInput[(row * width + col) * 3 + RED] * 0.299 + imgInput[(row * width + col) * 3 + GREEN] * 0.587 + imgInput[(row * width + col) * 3 + BLUE] * 0.114;
        }
    }
    
    //return *imgOutput; // No es necesario retornar ya que es un puntero.
}

unsigned char clamp(int value)
{
    if(value < 0)
        value = 0;
    else
        if(value > 255)
            value = 255;
    return (unsigned char)value;
}

void sobelFilter(unsigned char *imgOutput, int maskWidth, char *M, unsigned char *imgInput, int width, int height)
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
            for (int i = 0; i < maskWidth; i++)
            {
                for (int j = 0; j < maskWidth; j++)
                {
                    if ((N_start_point_col + j >= 0 && N_start_point_col + j < width) && (N_start_point_row + i >= 0 && N_start_point_row + i < height))
                    {
                        Pvalue += imgOutput[(N_start_point_row + i) * width + (N_start_point_col + j)];
                    }
                }
            }
        }
    }

    imgOutput[row * width + col] = clamp(Pvalue);
    //return *imgOutput;  // No es necesario retornar ya que es un puntero.
}


int main(int argc, char **argv)
{
    char *imageName = argv[1];
    char M[] = {-1,0,1,-2,0,2,-1,0,1};
    unsigned char *dataRawImage, *imgOutput;
    unsigned char *sobelOutput;
    
    Mat image;
    image = imread(imageName, 1);
  
    Size img_size = image.size();

    int width = img_size.width;
    int height = img_size.height;
    int size = sizeof(unsigned char) * width * height * image.channels();
    int sizeGray = sizeof(unsigned char) * width * height;

    dataRawImage = (unsigned char*)malloc(size);
    imgOutput = (unsigned char*)malloc(sizeGray);
    sobelOutput = (unsigned char*)malloc(sizeGray);

    dataRawImage = image.data;

    img2gray(imgOutput, dataRawImage, width, height);

    Mat gray_image, grad_x, abs_grad_x;
    gray_image.create(height,width,CV_8UC1);
    gray_image.data = imgOutput;

	sobelFilter(sobelOutput, 3, M, imgOutput, width, height);

    Mat sobel_image;
    sobel_image.create(height,width,CV_8UC1);
    sobel_image.data = sobelOutput;

    namedWindow(imageName, WINDOW_NORMAL);
    namedWindow("Gray Image Secuential", WINDOW_NORMAL);
    namedWindow("Sobel Image OpenCV", WINDOW_NORMAL);

    imshow(imageName,image);
    imshow("Gray Image Secuential", gray_image);
    imshow("Sobel Image OpenCV", sobel_image);

    waitKey(0);

    free(dataRawImage);
    free(imgOutput);
    free(sobelOutput);
    
    return 0;
}
