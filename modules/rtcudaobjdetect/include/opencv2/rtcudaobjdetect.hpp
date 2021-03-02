/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef OPENCV_RTCUDAOBJDETECT_HPP
#define OPENCV_RTCUDAOBJDETECT_HPP

#ifndef __cplusplus
#  error cudaobjdetect.hpp header must be compiled as C++
#endif

#include "opencv2/core/cuda.hpp"
#include <pgm.h>
#include <litmus.h>
#include <vector>
#include <cuda_runtime.h>

/* Include the LITMUS^RT user space library header.
 * This header, part of liblitmus, provides the user space API of
 * LITMUS^RT.
 */
#include <litmus.h>
/* LITMUS^RT */

# define UNUSED(x) UNUSED_ ## x __attribute__((unused))

enum scheduling_option
{
    END_TO_END = 0,
    COARSE_GRAINED,      // 1
    FINE_GRAINED,        // 2
    COARSE_UNROLLED,     // 3
    CONFIGURABLE,        // 4
    SCHEDULING_OPTION_END
};

enum node_config
{
    NODE_INVALID = 0,
    NODE_A,     //  1
    NODE_B,     //  2
    NODE_C,     //  3
    NODE_D,     //  4
    NODE_E,     //  5
    NODE_AB,    //  6
    NODE_BC,    //  7
    NODE_CD,    //  8
    NODE_DE,    //  9
    NODE_ABC,   // 10
    NODE_BCD,   // 11
    NODE_CDE,   // 12
    NODE_ABCD,  // 13
    NODE_BCDE,  // 14
    NODE_ABCDE, // 15
    NODE_NONE   // 16
};

struct sync_info
{
    unsigned long long start_time;
    int job_no;
};

struct task_info
{
    int id;        // index of the node within the graph
    int graph_idx; // index of graph (0 to parallelism-1)
    /* real-time parameters in milliseconds */
    float period;
    float phase;
    /* fair-lateness priority point */
    float relative_deadline;
    int cluster;
    enum scheduling_option sched;
    bool realtime;
    bool early;
    struct sync_info *s_info_in;
    struct sync_info *s_info_out;
    std::vector<node_config> *source_config;
    std::vector<node_config> *sink_config;
    bool has_display_node;
};

/* Next, we define period and execution cost to be constant.
 * These are only constants for convenience in this example, they can be
 * determined at run time, e.g., from command line parameters.
 *
 * These are in milliseconds.
 */
#define PERIOD            62.5
#define RELATIVE_DEADLINE 62.5
#define EXEC_COST         5
/**
  @addtogroup cuda
  @{
      @defgroup cudaobjdetect Object Detection
  @}
 */

namespace cv { namespace cuda {

//! @addtogroup cudaobjdetect
//! @{

//
// HOG_RT (Histogram-of-Oriented-Gradients) Descriptor and Object Detector
//

/** @brief The class implements Histogram of Oriented Gradients (@cite Dalal2005) object detector.

@note
    -   Uses real-time scheduling with LITMUS^RT
    -   A CUDA example applying the Real-Time HOG descriptor for people detection can be found at
        opencv_source_code/samples/rt/gpu_hog.cpp
 */
class CV_EXPORTS HOG_RT : public Algorithm
{
public:

    enum
    {
        DESCR_FORMAT_ROW_BY_ROW,
        DESCR_FORMAT_COL_BY_COL
    };

    /** @brief Creates the HOG descriptor and detector.

    @param win_size Detection window size. Align to block size and block stride.
    @param block_size Block size in pixels. Align to cell size. Only (16,16) is supported for now.
    @param block_stride Block stride. It must be a multiple of cell size.
    @param cell_size Cell size. Only (8, 8) is supported for now.
    @param nbins Number of bins. Only 9 bins per cell are supported for now.
     */
    static Ptr<HOG_RT> create(Size win_size = Size(64, 128),
                              Size block_size = Size(16, 16),
                              Size block_stride = Size(8, 8),
                              Size cell_size = Size(8, 8),
                              int nbins = 9);

    //! Gaussian smoothing window parameter.
    virtual void setWinSigma(double win_sigma) = 0;
    virtual double getWinSigma() const = 0;

    //! L2-Hys normalization method shrinkage.
    virtual void setL2HysThreshold(double threshold_L2hys) = 0;
    virtual double getL2HysThreshold() const = 0;

    //! Flag to specify whether the gamma correction preprocessing is required or not.
    virtual void setGammaCorrection(bool gamma_correction) = 0;
    virtual bool getGammaCorrection() const = 0;

    //! Maximum number of detection window increases.
    virtual void setNumLevels(int nlevels) = 0;
    virtual int getNumLevels() const = 0;

    //! Threshold for the distance between features and SVM classifying plane.
    //! Usually it is 0 and should be specified in the detector coefficients (as the last free
    //! coefficient). But if the free coefficient is omitted (which is allowed), you can specify it
    //! manually here.
    virtual void setHitThreshold(double hit_threshold) = 0;
    virtual double getHitThreshold() const = 0;

    //! Window stride. It must be a multiple of block stride.
    virtual void setWinStride(Size win_stride) = 0;
    virtual Size getWinStride() const = 0;

    //! Coefficient of the detection window increase.
    virtual void setScaleFactor(double scale0) = 0;
    virtual double getScaleFactor() const = 0;

    //! Coefficient to regulate the similarity threshold. When detected, some
    //! objects can be covered by many rectangles. 0 means not to perform grouping.
    //! See groupRectangles.
    virtual void setGroupThreshold(int group_threshold) = 0;
    virtual int getGroupThreshold() const = 0;

    //! Descriptor storage format:
    //!   - **DESCR_FORMAT_ROW_BY_ROW** - Row-major order.
    //!   - **DESCR_FORMAT_COL_BY_COL** - Column-major order.
    virtual void setDescriptorFormat(int descr_format) = 0;
    virtual int getDescriptorFormat() const = 0;

    /** @brief Returns the number of coefficients required for the classification.
     */
    virtual size_t getDescriptorSize() const = 0;

    /** @brief Returns the block histogram size.
     */
    virtual size_t getBlockHistogramSize() const = 0;

    /** @brief Sets coefficients for the linear SVM classifier.
     */
    virtual void setSVMDetector(InputArray detector) = 0;

    /** @brief Returns coefficients of the classifier trained for people detection.
     */
    virtual Mat getDefaultPeopleDetector() const = 0;

    /* fine-grained nodes */
    virtual void* thread_fine_compute_scales(node_t node, pthread_barrier_t* init_barrier, struct task_info t_info) = 0;
    virtual void* thread_fine_resize(node_t node, pthread_barrier_t* init_barrier, struct task_info t_info) = 0;
    virtual void* thread_fine_compute_gradients(node_t node, pthread_barrier_t* init_barrier, struct task_info t_info) = 0;
    virtual void* thread_fine_compute_histograms(node_t node, pthread_barrier_t* init_barrier, struct task_info t_info) = 0;
    virtual void* thread_fine_normalize_histograms(node_t node, pthread_barrier_t* init_barrier, struct task_info t_info) = 0;
    virtual void* thread_fine_classify(node_t node, pthread_barrier_t* init_barrier, struct task_info t_info) = 0;
    virtual void* thread_fine_collect_locations(node_t node, pthread_barrier_t* init_barrier, struct task_info t_info) = 0;

    /* two-node intra-level combinations */
    virtual void* thread_fine_AB(node_t node, pthread_barrier_t* init_barrier, struct task_info t_info) = 0; // resize          + compute grads
    virtual void* thread_fine_BC(node_t node, pthread_barrier_t* init_barrier, struct task_info t_info) = 0; // compute grads   + compute hists
    virtual void* thread_fine_CD(node_t node, pthread_barrier_t* init_barrier, struct task_info t_info) = 0; // compute hists   + normalize hists
    virtual void* thread_fine_DE(node_t node, pthread_barrier_t* init_barrier, struct task_info t_info) = 0; // normalize hists + classify hists

    /* three-node intra-level combinations */
    virtual void* thread_fine_ABC(node_t node, pthread_barrier_t* init_barrier, struct task_info t_info) = 0; // resize        -> compute hists
    virtual void* thread_fine_BCD(node_t node, pthread_barrier_t* init_barrier, struct task_info t_info) = 0; // compute grads -> normalize hists
    virtual void* thread_fine_CDE(node_t node, pthread_barrier_t* init_barrier, struct task_info t_info) = 0; // compute hists -> classify hists

    /* four-node intra-level combinations */
    virtual void* thread_fine_ABCD(node_t node, pthread_barrier_t* init_barrier, struct task_info t_info) = 0; // resize        -> normalize hists
    virtual void* thread_fine_BCDE(node_t node, pthread_barrier_t* init_barrier, struct task_info t_info) = 0; // compute grads -> classify hists

    /* five-node entire-level combination */
    virtual void* thread_fine_ABCDE(node_t node, pthread_barrier_t* init_barrier, struct task_info t_info) = 0; // resize -> classify hists

    /* color-convert and source-node combination */
    virtual void fine_CC_S_ABCDE(struct task_info &t_info, void** out_buf_ptrs,
                                 cuda::GpuMat* gpu_img,
                                 cuda::GpuMat** grad_array, cuda::GpuMat** qangle_array,
                                 cuda::GpuMat** block_hists_array,
                                 cuda::GpuMat** smaller_img_array, cuda::GpuMat** labels_array,
                                 std::vector<Rect>* found,
                                 Mat *img, int frame_idx, const cudaStream_t& stream, lt_t frame_start_time,
                                 int omlp_sem_od) = 0; // color-convert -> classify hists (maybe not all the way)

    /* source-node combinations */
    virtual void* thread_fine_S_A(node_t node, pthread_barrier_t* init_barrier, struct task_info t_info) = 0;     // compute-levels +  resize
    virtual void* thread_fine_S_AB(node_t node, pthread_barrier_t* init_barrier, struct task_info t_info) = 0;    // compute-levels -> compute grads
    virtual void* thread_fine_S_ABC(node_t node, pthread_barrier_t* init_barrier, struct task_info t_info) = 0;   // compute-levels -> compute hists
    virtual void* thread_fine_S_ABCD(node_t node, pthread_barrier_t* init_barrier, struct task_info t_info) = 0;  // compute-levels -> normalize hists
    virtual void* thread_fine_S_ABCDE(node_t node, pthread_barrier_t* init_barrier, struct task_info t_info) = 0; // compute-levels -> classify hists

    /* sink-node combinations */
    virtual void* thread_fine_ABCDE_T(node_t node, pthread_barrier_t* init_barrier, struct task_info t_info) = 0; // resize          -> collect-locations
    virtual void* thread_fine_BCDE_T(node_t node, pthread_barrier_t* init_barrier, struct task_info t_info) = 0;  // compute grads   -> collect-locations
    virtual void* thread_fine_CDE_T(node_t node, pthread_barrier_t* init_barrier, struct task_info t_info) = 0;   // compute hists   -> collect-locations
    virtual void* thread_fine_DE_T(node_t node, pthread_barrier_t* init_barrier, struct task_info t_info) = 0;    // normalize hists -> collect-locations
    virtual void* thread_fine_E_T(node_t node, pthread_barrier_t* init_barrier, struct task_info t_info) = 0;     // classify hists   + collect-locations

    virtual void set_up_litmus_task(const struct task_info &t_info, struct rt_task &param, int *sem_od) = 0;
    virtual void set_up_constants(const cudaStream_t& stream) = 0;

    virtual int open_lock(int resource_id) = 0;
    virtual int lock_fzlp(int sem_od) = 0;
    virtual int wait_forbidden_zone(int sem_od, node_config computation) = 0;
    virtual int exit_forbidden_zone(int sem_od) = 0;
    virtual int unlock_fzlp(int sem_od) = 0;

    virtual int getTotalHistSize(Size img_size) const = 0;

    virtual int numPartsWithin(int size, int part_size, int stride) const = 0;
    virtual Size numPartsWithin(Size size, Size part_size, Size stride) const = 0;
};

//! @}

}} // namespace cv { namespace cuda {

#endif /* OPENCV_RTCUDAOBJDETECT_HPP */
