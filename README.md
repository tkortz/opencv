## Experiments with HOG

We compared the response times of successively finer-grained notions of DAG
scheduling, corresponding to monolithic, coarse-grained, and fine-grained HOG
DAGs. Please see more details in
[paper](https://cs.unc.edu/~anderson/papers/rtss18b.pdf).

### Requirements

Our implementation is based on LITMUS^RT, liblitmus, and PGM^RT, for which our
code have not been upstreamed yet. So please find forked repositories of these
projects in my GitHub profile for [liblitmus](https://github.com/Yougmark/liblitmus/tree/rtss18-gpu-wip) and [LITMUS^RT](https://github.com/Yougmark/litmus-rt/tree/rtss18-gpu-wip).

Our experiments were conducted with CUDA 9.1 and NVIDIA Driver 391 on Ubuntu
16.04 with the kernel LITMUS^RT 2017.1.

### Compilation

1. First, clone this repository and checkout rtss18 branch.

2. To compile opencv with liblitmus and PGM^RT, please modify CMakeLists.txt
accordingly so required header files and libraries can be found.

```
set(OPENCV_LINKER_LIBS ${OPENCV_LINKER_LIBS} pgm boost_graph boost_filesystem boost_system)
ocv_include_directories("/home/ming/pgm/include")  						# change this to your path for pgm header file
ocv_include_directories("/home/ming/litmus/liblitmus/include")  				# change this to your path for liblitmus header file
ocv_include_directories("/home/ming/litmus/liblitmus/arch/x86/include")  			# and this
ocv_include_directories("/home/ming/litmus/liblitmus/arch/x86/include/uapi")   			# and this
ocv_include_directories("/home/ming/litmus/liblitmus/arch/x86/include/generated/uapi") 		# and this
link_directories("/home/ming/pgm/")  								# change this to your path for pgm shared library file
link_directories("/home/ming/litmus/liblitmus/")  						# change this to your path for liblitmus shared library file
```

3. Follow [opencv document](https://docs.opencv.org/master/d7/d9f/tutorial_linux_install.html) to get required packages.

4. Make build directory and get into it: `mkdir build && cd build`

5. Prepare makefile with cmake: `cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local -D WITH_CUDA=ON -D BUILD_EXAMPLES=Yes -D ENABLE_CXX11=Yes ..`

6. Compile HOG: `make example_gpu_hog`

### Usage

```
Usage: example_gpu_hog
--video <video> 		# frames source
[--sched <int>] 		# scheduling option (0:end_to_end, 1:coarse_grained (same as 3 now), 2:fine_grained, 3:unrolled_coarse_grained)
[--count <int>] 		# num of frames to process
[--graph_bound <int>]		# response time bound of fine-grained HOG
[--cluster <int>] 		# cluster ID of this task
[--id <int>] 			# task ID of this task
[--rt <true/false>] 		# run under LITMUS^RT scheduler or not
[--display <true/false>] 	# to display result frame or not
```

We used this [video](https://github.com/opencv/opencv_extra/blob/master/testdata/gpu/video/768x576.avi) provided by opencv.

---

Below is the original content of opencv readme.

## OpenCV: Open Source Computer Vision Library

### Resources

* Homepage: <http://opencv.org>
* Docs: <http://docs.opencv.org/master/>
* Q&A forum: <http://answers.opencv.org>
* Issue tracking: <https://github.com/opencv/opencv/issues>

### Contributing

Please read the [contribution guidelines](https://github.com/opencv/opencv/wiki/How_to_contribute) before starting work on a pull request.

#### Summary of the guidelines:

* One pull request per issue;
* Choose the right base branch;
* Include tests and documentation;
* Clean up "oops" commits before submitting;
* Follow the [coding style guide](https://github.com/opencv/opencv/wiki/Coding_Style_Guide).
