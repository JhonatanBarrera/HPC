cmake_minimum_required(VERSION 2.8)
project( ImageConvolutionSobel )
find_package( OpenCV REQUIRED )
add_executable( secuSobel secuSobel.cpp )
target_link_libraries( secuSobel ${OpenCV_LIBS} )

