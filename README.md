### Requirements

Our implementation is based on LITMUS^RT, liblitmus, and PGM^RT (if using multiple nodes in the HOG graph).  The repositories/patches for the versions used are here:

* [Modified LITMUS^RT](https://www.cs.unc.edu/~tamert/papers/timewall_litmusrt.patch), against the [5.4.0-rc7 LITMUSRT kernel](https://github.com/JoshuaJB/litmus-rt/tree/linux-5.4-litmus) (commit 55ce62849)
* [Modified liblitmus](https://www.cs.unc.edu/~tamert/papers/timewall_liblitmus.patch), against the [base Liblitmus repository](https://github.com/LITMUS-RT/liblitmus) (commit a430c7b5)
* [PGM^RT](https://github.com/GElliott/pgm)

Our experiments were conducted with CUDA 10.2 and NVIDIA Driver version 440.33 on Ubuntu 18.04.

### Compilation

1. First, clone this repository and checkout the `timewall` branch.

2. Configure environment variables that point to library directories:
   ```
   # liblitmus directory
   export LIBLITMUS_DIR=

   # PGM^RT directory
   export PGM_RT_DIR=
   ```

3. Follow [opencv documentation](https://docs.opencv.org/master/d7/d9f/tutorial_linux_install.html) to get required packages.

4. Make the build directory and navigate into it: `mkdir build && cd build`.

5. Prepare the makefile with cmake.  We used this command:
   ```
   cmake -D CMAKE_BUILD_TYPE=Release \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D BUILD_EXAMPLES=Yes \
      -D ENABLE_PRECOMPILED_HEADERS=OFF \
      -D ENABLE_CXX11=Yes \
      -D BUILD_opencv_cudacodec=OFF \
      -D WITH_CUDA=ON \
      -D CUDA_GENERATION=Volta \
      -D CUDA_ARCH_BIN=7.0 \
      -D WITH_GSTREAMER_0_10=ON \
      ..
   ```

   Use `-D CUDA_GENERATION=Name` to select CUDA generation it will compile for, e.g., Volta.

6. Compile HOG (use multiple cores to make it go faster): `make -j8 example_rt_gpu_hog`.

### Usage

```
Usage: example_gpu_hog
--video <video>              # frames source
[--sched <int>]              # scheduling option (4:configurable-nodes)
[--level_config_file <file>] # file specifying the graph configuration for configurable-nodes scheduling
[--count <int>]              # num of frames to process
[--graph_bound <int>]        # end-to-end response-time bound of HOG
[--id <int>]                 # task ID of this task (e.g., 1 or 2)
[--rt <true/false>]          # whether to run under LITMUS^RT scheduler
[--display <true/false>]     # whether to display the resulting frame
[--make_gray <true/false>]   # work with grayscale or color images
[--display <true/false>]     # whether to display (should be true)
```

We used this [video](https://github.com/opencv/opencv_extra/blob/master/testdata/gpu/video/768x576.avi) provided by opencv.  Note that we used the EXT-RES (TimeWall) scheduler under LITMUS^RT, and we ran one or three instances of the program together to simulate processing of multiple different camera inputs.  With a parallelism level of `2` and a frame count of `5000`, we ran the program as follows (where `$i` corresponds to each of the instances we ran together, and `$file` points to the graph configuration being tested):
```
example_gpu_hog
--video ~/opencv_extra/testdata/gpu/video/768x576.avi
--sched 4
--level_config_file $file
--count 5000
--graph_bound 80
--id $i
--rt true
--display false
--make_gray true
---
```

Below is the original content of the OpenCV readme.

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
