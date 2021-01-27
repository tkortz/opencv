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
  opencv_videoio
  opencv_highgui
  opencv_video
  opencv_videostab
  pgm
  litmus)
ocv_check_dependencies(${OPENCV_RT_SAMPLES_REQUIRED_DEPS})

if(NOT BUILD_EXAMPLES OR NOT OCV_DEPENDENCIES_FOUND)
  return()
endif()

project(rt_samples)
ocv_include_modules_recurse(${OPENCV_RT_SAMPLES_REQUIRED_DEPS})
file(GLOB_RECURSE rt_samples RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)

foreach(sample_filename ${rt_samples})
  set(package "rt")
  ocv_define_sample(tgt ${sample_filename} ${package})
  ocv_target_link_libraries(${tgt} ${OPENCV_LINKER_LIBS} ${OPENCV_RT_SAMPLES_REQUIRED_DEPS})
endforeach()