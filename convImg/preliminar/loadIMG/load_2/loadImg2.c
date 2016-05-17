#include <cv.h>
#include <highgui.h>

int main(int argc, char **argv)
{
    char* imageName = argv[1];
    
    Mat image;
    image = imread(imageName, 1);

    namedWindow(imageName, WINDOW_NORMAL);

    imshow(imageName,image);
    
    waitKey(0);
    
    return 0;
}
