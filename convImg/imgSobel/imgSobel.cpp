#include <cv.h>
#include <highgui.h>

using namespace cv;

int main(int argc, char **argv)
{
    char *imageName = argv[1];
    
    Mat image, imgSobel;
    image = imread(imageName, 1);
    
    GaussianBlur( image, image, Size(3,3), 0, 0, BORDER_DEFAULT );
  
    Mat gray_image_opencv, grad_x, grad_y, abs_grad_x, abs_grad_y;
    
    cvtColor(image, gray_image_opencv, CV_BGR2GRAY);
    
    Sobel(gray_image_opencv,grad_x,CV_16S,1,0,3,1,0,BORDER_DEFAULT);
    convertScaleAbs(grad_x, abs_grad_x);
    
    Sobel(gray_image_opencv,grad_y,CV_16S,1,0,3,1,0,BORDER_DEFAULT);
    convertScaleAbs(grad_y, abs_grad_y);
    
    /// Total Gradient (approximate)
  	addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, imgSobel );
		
    namedWindow(imageName, WINDOW_NORMAL);
    namedWindow("Gray Image OpenCV", WINDOW_NORMAL);

    imshow(imageName,image);
    imshow("Gray Image OpenCV",imgSobel);

    waitKey(0);
    
    return 0;
}
