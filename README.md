## Experiments with HOG


### Requirements


### Compilation

1. First, clone this repository and checkout `fzlp` branch.

2. Configure environment variables that point to library directories:
   ```
   # liblitmus directory
   export LIBLITMUS_DIR=

   # PGM^RT directory
   export PGM_RT_DIR=

   # FZLP directory
   export FZLP_DIR=
   ```

3. Follow [opencv document](https://docs.opencv.org/master/d7/d9f/tutorial_linux_install.html) to get required packages.

4. Make build directory and get into it: `mkdir build && cd build`

5. Prepare makefile with cmake: `cmake -D CMAKE_BUILD_TYPE=Release -D
   CMAKE_INSTALL_PREFIX=/usr/local -D WITH_CUDA=ON -D BUILD_EXAMPLES=Yes -D
   ENABLE_CXX11=Yes -D BUILD_opencv_cudacodec=OFF ..`.  Use `-D
   CUDA_GENERATION=Name` to select CUDA generation it will compile for, e.g.,
   Volta.

6. Compile HOG: `make example_gpu_hog`

### Usage

> To be updated:

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
