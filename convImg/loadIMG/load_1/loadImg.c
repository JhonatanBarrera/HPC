#include <opencv/highgui.h>

int main( int argc, char** argv )
{
    IplImage* img = cvLoadImage(argv[1],CV_LOAD_IMAGE_UNCHANGED ); 

    cvNamedWindow( "ImagenCargada", CV_WINDOW_AUTOSIZE ); 
    cvShowImage("ImagenCargada", img );

    cvWaitKey(0);

    cvReleaseImage( &img );

    cvDestroyWindow("ImagenCargada" );
}
