#include <cv.h>
#include <highgui.h>

using namespace cv;

int main(int argc, char **argv)
{
    char *imageName = argv[1];
    unsigned char *dataRawImage;
    
    Mat image;
    image = imread(imageName, 1);
    
    Size sizeimg = image.size();
  
    Mat gray_image_opencv, grad_x;
    cvtColor(image, gray_image_opencv, CV_BGR2GRAY);
    Sobel(gray_image_opencv,grad_x,CV_8UC1,1,0,3,1,0,BORDER_DEFAULT);
		
    namedWindow(imageName, WINDOW_NORMAL);
    namedWindow("Gray Image OpenCV", WINDOW_NORMAL);

    imshow(imageName,image);
    imshow("Gray Image OpenCV",gray_image_opencv);

    waitKey(0);
    
    return 0;
}
