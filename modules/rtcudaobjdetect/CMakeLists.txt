ocv_include_directories($ENV{PGM_RT_DIR}/include)
ocv_include_directories($ENV{LIBLITMUS_DIR}/include)
ocv_include_directories($ENV{LIBLITMUS_DIR}/arch/x86/include)
ocv_include_directories($ENV{LIBLITMUS_DIR}/arch/x86/include/uapi)
ocv_include_directories($ENV{LIBLITMUS_DIR}/arch/x86/include/generated/uapi)

link_directories($ENV{PGM_RT_DIR})
link_directories($ENV{LIBLITMUS_DIR})

set(OPENCV_LINKER_LIBS ${OPENCV_LINKER_LIBS} pgm)

if(IOS OR (NOT HAVE_CUDA AND NOT BUILD_CUDA_STUBS))
  ocv_module_disable(rtcudaobjdetect)
endif()

set(the_description "CUDA-accelerated Object Detection")

ocv_warnings_disable(CMAKE_CXX_FLAGS /wd4127 /wd4324 /wd4512 -Wundef -Wmissing-declarations -Wshadow)

ocv_define_module(rtcudaobjdetect opencv_objdetect opencv_cudaarithm opencv_cudawarping OPTIONAL opencv_cudalegacy)
