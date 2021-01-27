ocv_install_example_src(rt *.cpp *.hpp CMakeLists.txt)

ocv_include_directories($ENV{PGM_RT_DIR}/include)
ocv_include_directories($ENV{LIBLITMUS_DIR}/include)
ocv_include_directories($ENV{LIBLITMUS_DIR}/arch/x86/include)
ocv_include_directories($ENV{LIBLITMUS_DIR}/arch/x86/include/uapi)
ocv_include_directories($ENV{LIBLITMUS_DIR}/arch/x86/include/generated/uapi)

link_directories($ENV{PGM_RT_DIR})
link_directories($ENV{LIBLITMUS_DIR})

set(OPENCV_RT_SAMPLES_REQUIRED_DEPS
  opencv_core
  opencv_imgproc
  opencv_imgcodecs
  opencv_videoio
  opencv_highgui
  opencv_objdetect
  opencv_video
  opencv_videostab
  opencv_cudaarithm
  opencv_cudawarping
  opencv_cudaimgproc
  opencv_cudalegacy
  opencv_cudaobjdetect
  pgm
  litmus)
ocv_check_dependencies(${OPENCV_RT_SAMPLES_REQUIRED_DEPS})

if(NOT BUILD_EXAMPLES OR NOT OCV_DEPENDENCIES_FOUND)
  return()
endif()

project(rt_samples)

if(COMMAND ocv_warnings_disable)
  ocv_warnings_disable(CMAKE_CXX_FLAGS -Wsuggest-override -Winconsistent-missing-override)
endif()

ocv_include_modules_recurse(${OPENCV_RT_SAMPLES_REQUIRED_DEPS})

if(HAVE_CUDA)
  ocv_include_directories(${CUDA_INCLUDE_DIRS})
endif()

if((CV_GCC OR CV_CLANG) AND NOT ENABLE_NOISY_WARNINGS)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-unused-function")
endif()

file(GLOB_RECURSE rt_samples RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)

foreach(sample_filename ${rt_samples})
  set(package "rt")
  ocv_define_sample(tgt ${sample_filename} ${package})
  ocv_target_link_libraries(${tgt} ${OPENCV_LINKER_LIBS} ${OPENCV_RT_SAMPLES_REQUIRED_DEPS})
  if(HAVE_CUDA AND NOT ANDROID)
    ocv_target_link_libraries(${tgt} ${CUDA_CUDA_LIBRARY})
  endif()
endforeach()