## Experiments with configurable-node HOG

We compared the response times of different variants of HOG, in which the granularity of schedulable entities changes.  This can be done by starting with the fine-grained version of HOG (see https://github.com/Yougmark/opencv) and merging nodes together.

### Requirements

As with [prior work](https://cs.unc.edu/~anderson/papers/rtss18b.pdf), our implementation is based on LITMUS^RT, liblitmus, and PGM^RT, as previously modified in the prior work.  The repositories for these modified version are here:

* [Modified LITMUS^RT](https://github.com/Yougmark/litmus-rt/tree/rtss18-gpu-wip)
* [Modified liblitmus](https://github.com/Yougmark/liblitmus/tree/rtss18-gpu-wip)
* [PGM^RT](https://github.com/GElliott/pgm)

Our experiments were conducted with CUDA 10.2 and NVIDIA Driver 440.33 on Ubuntu 16.04 with the kernel LITMUS^RT 2017.1.

### Compilation

1. First, clone this repository and checkout the `configurable-nodes-submission` branch.

2. Configure environment variables that point to library directories:
   ```
   # liblitmus directory
   export LIBLITMUS_DIR=

   # PGM^RT directory
   export PGM_RT_DIR=
   ```

3. Follow [opencv documentation](https://docs.opencv.org/master/d7/d9f/tutorial_linux_install.html) to get required packages.

4. Make the build directory and navigate into it: `mkdir build && cd build`

5. Prepare makefile with cmake:
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

   Use `-D CUDA_GENERATION=Name` to select CUDA generation it will compile for, e.g.,
   Volta.

6. Compile HOG: `make example_gpu_hog`

### Usage

```
Usage: example_gpu_hog
--video <video>              # frames source
[--sched <int>]              # scheduling option (0:end-to-end, 1:coarse-grained, 2:fine-grained, 3:unrolled-coarse-grained, 4:configurable-nodes)
[--level_config_file <file>] # file specifying the graph configuration for configurable-nodes scheduling (defaults to fine-grained)
[--count <int>]              # num of frames to process
[--graph_bound <int>]        # response time bound of fine-grained HOG
[--cluster <int>]            # cluster ID of this task
[--id <int>]                 # task ID of this task
[--rt <true/false>]          # whether to run under LITMUS^RT scheduler
[--display <true/false>]     # whether to display the resulting frame
[--make_gray <true/false>]   # work with grayscale or color images
```

We used this [video](https://github.com/opencv/opencv_extra/blob/master/testdata/gpu/video/768x576.avi) provided by opencv.  Note that we used the GSN-EDF scheduler under LITMUS^RT, and we ran six instances of the program together to simulate processing of six different camera inputs.  With a parallelism level of `2` and a frame count of `5000`, we ran the program as follows (where `$i` corresponds to each of the six instances we ran together, and `$file` points to the graph configuration being tested):
```
example_gpu_hog
--video ~/opencv_extra/testdata/gpu/video/768x576.avi
--sched 4
--level_config_file $file
--count 5000
--graph_bound 66
--id $i
--rt true
--display false
--make_gray true
---
```

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
