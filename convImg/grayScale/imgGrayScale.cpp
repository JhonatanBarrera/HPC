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
    
    int width = sizeimg.width;
    int height = sizeimg.height;
    int sizeCh = sizeof(unsigned char) * width * height * image.channels();;
    int sizeGray = sizeof(unsigned char) * width * height;

		Mat gray_image;
    gray_image.create(height,width,CV_8UC1);
    
    Mat gray_image_opencv;
    cvtColor(image, gray_image_opencv, CV_BGR2GRAY);
		
    namedWindow(imageName, WINDOW_NORMAL);
    namedWindow("Gray Image OpenCV", WINDOW_NORMAL);

    imshow(imageName,image);
    imshow("Gray Image OpenCV",gray_image_opencv);

    waitKey(0);
    
    return 0;
}
