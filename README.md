## Tracking by Detection

We added a sample to OpenCV to demonstrate a tracking-by-detection (TBD) application.  This was used for evaluation in [our paper](https://cs.unc.edu/~anderson/papers/isorc20.pdf), presented at ISORC '20.

### Compilation

1. First, clone this repository and checkout the `acc-vs-hist` branch.

2. Follow [opencv documentation](https://docs.opencv.org/master/d7/d9f/tutorial_linux_install.html) to get required packages.

3. Make the build directory and navigate into it: `mkdir build && cd build`.

4. Prepare the makefile with cmake.  We used this command:
    ```
    cmake -D CMAKE_BUILD_TYPE=Release \
        -D CMAKE_INSTALL_PREFIX=/usr/local \
        -D BUILD_EXAMPLES=Yes \
        -D ENABLE_PRECOMPILED_HEADERS=OFF \
        -D ENABLE_CXX11=Yes \
        -D BUILD_opencv_cudacodec=OFF \
        -D WITH_CUDA=ON \
        -D CUDA_GENERATION=Volta \
        -D OPENCV_DNN_CUDA=ON \
        -D CUDA_ARCH_BIN=7.0 \
        -D WITH_OPENMP=ON \
        ..
    ```

   Use `-D CUDA_GENERATION=Name` to select the CUDA generation it will compile for, e.g., Volta.

5. Compile TBD (use multiple cores to make it go faster): `make -j8 example_gpu_tbd`.

### Usage

```
example_gpu_tbd
--folder <folder_path>)                 # load images from folder
--history_distribution <string>         # comma-separated distribution of age of
                                        #     history for tracking (e.g., '7,3'
                                        #     for 70% prior frame, 30% two prior)
--pedestrian_bbox_filename <string>     # filename of pedestrian bounding box
                                        #     results for ground truth
--vehicle_bbox_filename <string>        # filename of vehicle bounding box
                                        #     results for ground truth
--write_tracking <bool>                 # whether to output the tracking results
--pedestrian_tracking_filepath <string> # pedestrian-tracking output filename
--vehicle_tracking_filepath <string>]   # vehicle-tracking output filename
--num_tracking_iters <int>              # number of times to repeat the
                                        #     tracking experiment
--num_tracking_frames <int>             # number of frames of the video to track
```

If no bounding box file is provided for either of pedestrians or vehicles, they will not be tracked.

You can find overall instructions for reproducing our results in [our experiment-running repository](https://github.com/tkortz/isorc20_experiments), as well as additional scripts.

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
