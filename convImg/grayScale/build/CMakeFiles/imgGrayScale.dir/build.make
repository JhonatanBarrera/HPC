# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The program to use to edit the cache.
CMAKE_EDIT_COMMAND = /usr/bin/ccmake

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/hera_hpccourse/3mer/HPC/convImg/grayScale

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/hera_hpccourse/3mer/HPC/convImg/grayScale/build

# Include any dependencies generated for this target.
include CMakeFiles/imgGrayScale.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/imgGrayScale.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/imgGrayScale.dir/flags.make

CMakeFiles/imgGrayScale.dir/imgGrayScale.cpp.o: CMakeFiles/imgGrayScale.dir/flags.make
CMakeFiles/imgGrayScale.dir/imgGrayScale.cpp.o: ../imgGrayScale.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/hera_hpccourse/3mer/HPC/convImg/grayScale/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/imgGrayScale.dir/imgGrayScale.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/imgGrayScale.dir/imgGrayScale.cpp.o -c /home/hera_hpccourse/3mer/HPC/convImg/grayScale/imgGrayScale.cpp

CMakeFiles/imgGrayScale.dir/imgGrayScale.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/imgGrayScale.dir/imgGrayScale.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/hera_hpccourse/3mer/HPC/convImg/grayScale/imgGrayScale.cpp > CMakeFiles/imgGrayScale.dir/imgGrayScale.cpp.i

CMakeFiles/imgGrayScale.dir/imgGrayScale.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/imgGrayScale.dir/imgGrayScale.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/hera_hpccourse/3mer/HPC/convImg/grayScale/imgGrayScale.cpp -o CMakeFiles/imgGrayScale.dir/imgGrayScale.cpp.s

CMakeFiles/imgGrayScale.dir/imgGrayScale.cpp.o.requires:
.PHONY : CMakeFiles/imgGrayScale.dir/imgGrayScale.cpp.o.requires

CMakeFiles/imgGrayScale.dir/imgGrayScale.cpp.o.provides: CMakeFiles/imgGrayScale.dir/imgGrayScale.cpp.o.requires
	$(MAKE) -f CMakeFiles/imgGrayScale.dir/build.make CMakeFiles/imgGrayScale.dir/imgGrayScale.cpp.o.provides.build
.PHONY : CMakeFiles/imgGrayScale.dir/imgGrayScale.cpp.o.provides

CMakeFiles/imgGrayScale.dir/imgGrayScale.cpp.o.provides.build: CMakeFiles/imgGrayScale.dir/imgGrayScale.cpp.o

# Object files for target imgGrayScale
imgGrayScale_OBJECTS = \
"CMakeFiles/imgGrayScale.dir/imgGrayScale.cpp.o"

# External object files for target imgGrayScale
imgGrayScale_EXTERNAL_OBJECTS =

imgGrayScale: CMakeFiles/imgGrayScale.dir/imgGrayScale.cpp.o
imgGrayScale: CMakeFiles/imgGrayScale.dir/build.make
imgGrayScale: /usr/local/lib/libopencv_videostab.so.2.4.12
imgGrayScale: /usr/local/lib/libopencv_video.so.2.4.12
imgGrayScale: /usr/local/lib/libopencv_ts.a
imgGrayScale: /usr/local/lib/libopencv_superres.so.2.4.12
imgGrayScale: /usr/local/lib/libopencv_stitching.so.2.4.12
imgGrayScale: /usr/local/lib/libopencv_photo.so.2.4.12
imgGrayScale: /usr/local/lib/libopencv_ocl.so.2.4.12
imgGrayScale: /usr/local/lib/libopencv_objdetect.so.2.4.12
imgGrayScale: /usr/local/lib/libopencv_nonfree.so.2.4.12
imgGrayScale: /usr/local/lib/libopencv_ml.so.2.4.12
imgGrayScale: /usr/local/lib/libopencv_legacy.so.2.4.12
imgGrayScale: /usr/local/lib/libopencv_imgproc.so.2.4.12
imgGrayScale: /usr/local/lib/libopencv_highgui.so.2.4.12
imgGrayScale: /usr/local/lib/libopencv_gpu.so.2.4.12
imgGrayScale: /usr/local/lib/libopencv_flann.so.2.4.12
imgGrayScale: /usr/local/lib/libopencv_features2d.so.2.4.12
imgGrayScale: /usr/local/lib/libopencv_core.so.2.4.12
imgGrayScale: /usr/local/lib/libopencv_contrib.so.2.4.12
imgGrayScale: /usr/local/lib/libopencv_calib3d.so.2.4.12
imgGrayScale: /usr/lib/x86_64-linux-gnu/libGLU.so
imgGrayScale: /usr/lib/x86_64-linux-gnu/libGL.so
imgGrayScale: /usr/lib/x86_64-linux-gnu/libSM.so
imgGrayScale: /usr/lib/x86_64-linux-gnu/libICE.so
imgGrayScale: /usr/lib/x86_64-linux-gnu/libX11.so
imgGrayScale: /usr/lib/x86_64-linux-gnu/libXext.so
imgGrayScale: /usr/local/lib/libopencv_nonfree.so.2.4.12
imgGrayScale: /usr/local/lib/libopencv_ocl.so.2.4.12
imgGrayScale: /usr/local/lib/libopencv_gpu.so.2.4.12
imgGrayScale: /usr/local/lib/libopencv_photo.so.2.4.12
imgGrayScale: /usr/local/lib/libopencv_objdetect.so.2.4.12
imgGrayScale: /usr/local/lib/libopencv_legacy.so.2.4.12
imgGrayScale: /usr/local/lib/libopencv_video.so.2.4.12
imgGrayScale: /usr/local/lib/libopencv_ml.so.2.4.12
imgGrayScale: /usr/local/lib/libopencv_calib3d.so.2.4.12
imgGrayScale: /usr/local/lib/libopencv_features2d.so.2.4.12
imgGrayScale: /usr/local/lib/libopencv_highgui.so.2.4.12
imgGrayScale: /usr/local/lib/libopencv_imgproc.so.2.4.12
imgGrayScale: /usr/local/lib/libopencv_flann.so.2.4.12
imgGrayScale: /usr/local/lib/libopencv_core.so.2.4.12
imgGrayScale: /usr/local/cuda/lib64/libcudart.so
imgGrayScale: /usr/local/cuda/lib64/libnppc.so
imgGrayScale: /usr/local/cuda/lib64/libnppi.so
imgGrayScale: /usr/local/cuda/lib64/libnpps.so
imgGrayScale: /usr/local/cuda/lib64/libcufft.so
imgGrayScale: CMakeFiles/imgGrayScale.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable imgGrayScale"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/imgGrayScale.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/imgGrayScale.dir/build: imgGrayScale
.PHONY : CMakeFiles/imgGrayScale.dir/build

CMakeFiles/imgGrayScale.dir/requires: CMakeFiles/imgGrayScale.dir/imgGrayScale.cpp.o.requires
.PHONY : CMakeFiles/imgGrayScale.dir/requires

CMakeFiles/imgGrayScale.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/imgGrayScale.dir/cmake_clean.cmake
.PHONY : CMakeFiles/imgGrayScale.dir/clean

CMakeFiles/imgGrayScale.dir/depend:
	cd /home/hera_hpccourse/3mer/HPC/convImg/grayScale/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/hera_hpccourse/3mer/HPC/convImg/grayScale /home/hera_hpccourse/3mer/HPC/convImg/grayScale /home/hera_hpccourse/3mer/HPC/convImg/grayScale/build /home/hera_hpccourse/3mer/HPC/convImg/grayScale/build /home/hera_hpccourse/3mer/HPC/convImg/grayScale/build/CMakeFiles/imgGrayScale.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/imgGrayScale.dir/depend
