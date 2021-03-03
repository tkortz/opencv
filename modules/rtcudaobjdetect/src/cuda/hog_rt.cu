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

#if !defined CUDA_DISABLER

#include "opencv2/core/cuda/common.hpp"
#include "opencv2/core/cuda/reduce.hpp"
#include "opencv2/core/cuda/functional.hpp"
#include "opencv2/core/cuda/warp_shuffle.hpp"
#include "opencv2/rtcudaobjdetect.hpp"

#include <sys/types.h>
#include <unistd.h>

namespace cv { namespace cuda { namespace device
{

    namespace hog_rt
    {
        __constant__ int cnbins;
        __constant__ int cblock_stride_x;
        __constant__ int cblock_stride_y;
        __constant__ int cnblocks_win_x;
        __constant__ int cnblocks_win_y;
        __constant__ int cncells_block_x;
        __constant__ int cncells_block_y;
        __constant__ int cblock_hist_size;
        __constant__ int cblock_hist_size_2up;
        __constant__ int cdescr_size;
        __constant__ int cdescr_width;

        int use_locks = 1;

        /* resource_id must be a non-negative int */
        int open_fzlp_lock(int resource_id)
        {
            if (!use_locks) return -3;

            // fprintf(stdout, "[%d | %d] Attempting to open OMLP (%d) semaphore now.\n", gettid(), getpid(), OMLP_SEM);

            int lock_od = -1;
            obj_type_t protocol = OMLP_SEM;
            const char *lock_namespace = "./rtspin-locks";
            int cluster = 0;
            if (protocol >= 0) {
                /* open reference to semaphore */
                lock_od = litmus_open_lock(protocol, resource_id, lock_namespace, &cluster);
                if (lock_od < 0) {
                    perror("litmus_open_lock");
                    fprintf(stderr, "Could not open lock.\n");
                }
                // else {
                //     fprintf(stdout, "[%d | %d] Successfully opened OMLP semaphore lock: %d.\n", gettid(), getpid(), lock_od);
                // }
            }

            return lock_od;
        }

        int lock_fzlp(int sem_od)
        {
            if (!use_locks) return -3;

            int res = -2;

            if (sem_od >= 0)
            {
                // fprintf(stdout, "[%d | %d] Calling lock (%d) at time \t%llu\n", gettid(), getpid(), sem_od, litmus_clock());
                res = litmus_lock(sem_od);
                // fprintf(stdout, "[%d | %d] Acquired lock at time \t%llu (status=%d)\n", gettid(), getpid(), litmus_clock(), res);
            }

            return res;
        }

        int wait_forbidden_zone(int sem_od, node_config computation)
        {
            if (!use_locks) return -3;

            int res = -2;

            int zone_length = 0;
            int cpu_measured = 0;
            switch (computation) {
                case NODE_A: // resize
                    zone_length = us2ns(27);   // 27 us
                    cpu_measured = us2ns(141); // 141 us
                    break;
                case NODE_B: // compute gradients
                    zone_length = us2ns(42);   // 42 us
                    cpu_measured = us2ns(153); // 153 us
                    break;
                case NODE_C: // compute hists
                    zone_length = us2ns(56);   // 56 us
                    cpu_measured = us2ns(187); // 187 us
                    break;
                case NODE_D: // normalize hists
                    zone_length = us2ns(28);   // 28 us
                    cpu_measured = us2ns(145); // 145 us
                    break;
                case NODE_E: // classify hists
                    zone_length = us2ns(49);   // 49 us
                    cpu_measured = us2ns(174); // 174 us
                    break;
                case NODE_AB: // stand-in for copy-in of image
                    zone_length = us2ns(77);   // 77 us
                    cpu_measured = us2ns(197); // 197 us
                    break;
                case NODE_DE: // stand-in for copy-out of results
                    zone_length = us2ns(29);   // 29 us
                    cpu_measured = us2ns(49);  // 49 us
                    break;
                default:
                    zone_length = ms2ns(2);    // default to 2 milliseconds
                    cpu_measured = ms2ns(3);   // default to 3 milliseconds
                    break;
            }

            zone_length += us2ns(100);
            cpu_measured += us2ns(100);

            if (sem_od >= 0)
            {
                fprintf(stdout, "[%d | %d] Checking FZ at time \t%llu (computation %d)\n", gettid(), getpid(), litmus_clock(), computation);
                res = litmus_access_forbidden_zone_check(sem_od, cpu_measured, zone_length);
                fprintf(stdout, "[%d | %d] Not in FZ at time \t%llu (status=%d)\n", gettid(), getpid(), litmus_clock(), res);
            }

            return res;
        }

        int exit_forbidden_zone(int sem_od)
        {
            if (!use_locks) return -3;

            int res = -2;

            if (sem_od >= 0)
            {
                res = litmus_exit_forbidden_zone(sem_od);
            }

            return res;
        }

        int unlock_fzlp(int sem_od)
        {
            if (!use_locks) return -3;

            int res = -2;

            if (sem_od >= 0)
            {
                // fprintf(stdout, "[%d | %d] Unlocking at time \t\t%llu\n", gettid(), getpid(), litmus_clock());
                res = litmus_unlock(sem_od);
                // fprintf(stdout, "[%d | %d] Unlocked at time \t\t%llu (status=%d)\n", gettid(), getpid(), litmus_clock(), res);
            }

            return res;
        }

        /* Returns the nearest upper power of two, works only for
        the typical GPU thread count (pert block) values */
        int power_2up(unsigned int n)
        {
            if (n <= 1) return 1;
            else if (n <= 2) return 2;
            else if (n <= 4) return 4;
            else if (n <= 8) return 8;
            else if (n <= 16) return 16;
            else if (n <= 32) return 32;
            else if (n <= 64) return 64;
            else if (n <= 128) return 128;
            else if (n <= 256) return 256;
            else if (n <= 512) return 512;
            else if (n <= 1024) return 1024;
            return -1; // Input is too big
        }

        /* Returns the max size for nblocks */
        int max_nblocks(int nthreads, int ncells_block = 1)
        {
            int threads = nthreads * ncells_block;
            if(threads * 4 <= 256)
                return 4;
            else if(threads * 3 <= 256)
                return 3;
            else if(threads * 2 <= 256)
                return 2;
            else
                return 1;
        }


        void set_up_constants(int nbins,
                              int block_stride_x, int block_stride_y,
                              int nblocks_win_x, int nblocks_win_y,
                              int ncells_block_x, int ncells_block_y,
                              const cudaStream_t& stream)
        {
            cudaSafeCall(cudaMemcpyToSymbolAsync(cnbins,               &nbins,               sizeof(nbins),               0, cudaMemcpyHostToDevice, stream));
            cudaSafeCall(cudaMemcpyToSymbolAsync(cblock_stride_x,      &block_stride_x,      sizeof(block_stride_x),      0, cudaMemcpyHostToDevice, stream));
            cudaSafeCall(cudaMemcpyToSymbolAsync(cblock_stride_y,      &block_stride_y,      sizeof(block_stride_y),      0, cudaMemcpyHostToDevice, stream));
            cudaSafeCall(cudaMemcpyToSymbolAsync(cnblocks_win_x,       &nblocks_win_x,       sizeof(nblocks_win_x),       0, cudaMemcpyHostToDevice, stream));
            cudaSafeCall(cudaMemcpyToSymbolAsync(cnblocks_win_y,       &nblocks_win_y,       sizeof(nblocks_win_y),       0, cudaMemcpyHostToDevice, stream));
            cudaSafeCall(cudaMemcpyToSymbolAsync(cncells_block_x,      &ncells_block_x,      sizeof(ncells_block_x),      0, cudaMemcpyHostToDevice, stream));
            cudaSafeCall(cudaMemcpyToSymbolAsync(cncells_block_y,      &ncells_block_y,      sizeof(ncells_block_y),      0, cudaMemcpyHostToDevice, stream));

            int block_hist_size = nbins * ncells_block_x * ncells_block_y;
            cudaSafeCall(cudaMemcpyToSymbolAsync(cblock_hist_size,     &block_hist_size,     sizeof(block_hist_size),     0, cudaMemcpyHostToDevice, stream));

            int block_hist_size_2up = power_2up(block_hist_size);
            cudaSafeCall(cudaMemcpyToSymbolAsync(cblock_hist_size_2up, &block_hist_size_2up, sizeof(block_hist_size_2up), 0, cudaMemcpyHostToDevice, stream));

            int descr_width = nblocks_win_x * block_hist_size;
            cudaSafeCall(cudaMemcpyToSymbolAsync(cdescr_width,         &descr_width,         sizeof(descr_width),         0, cudaMemcpyHostToDevice, stream));

            int descr_size = descr_width * nblocks_win_y;
            cudaSafeCall(cudaMemcpyToSymbolAsync(cdescr_size,          &descr_size,          sizeof(descr_size),          0, cudaMemcpyHostToDevice, stream));
        }


        //----------------------------------------------------------------------------
        // Histogram computation
        //
        // CUDA kernel to compute the histograms
        template <int nblocks> // Number of histogram blocks processed by single GPU thread block
        __global__ void compute_hists_kernel_many_blocks(const int img_block_width, const PtrStepf grad,
                                                         const PtrStepb qangle, float scale, float* block_hists,
                                                         int cell_size, int patch_size, int block_patch_size,
                                                         int threads_cell, int threads_block, int half_cell_size)
        {
            const int block_x = threadIdx.z;
            const int cell_x = threadIdx.x / threads_cell;
            const int cell_y = threadIdx.y;
            const int cell_thread_x = threadIdx.x & (threads_cell - 1);

            if (blockIdx.x * blockDim.z + block_x >= img_block_width)
                return;

            extern __shared__ float smem[];
            float* hists = smem;
            float* final_hist = smem + cnbins * block_patch_size * nblocks;

            // patch_size means that patch_size pixels affect on block's cell
            if (cell_thread_x < patch_size)
            {
                const int offset_x = (blockIdx.x * blockDim.z + block_x) * cblock_stride_x +
                                     half_cell_size * cell_x + cell_thread_x;
                const int offset_y = blockIdx.y * cblock_stride_y + half_cell_size * cell_y;

                const float* grad_ptr = grad.ptr(offset_y) + offset_x * 2;
                const unsigned char* qangle_ptr = qangle.ptr(offset_y) + offset_x * 2;


                float* hist = hists + patch_size * (cell_y * blockDim.z * cncells_block_y +
                                            cell_x + block_x * cncells_block_x) +
                                           cell_thread_x;
                for (int bin_id = 0; bin_id < cnbins; ++bin_id)
                    hist[bin_id * block_patch_size * nblocks] = 0.f;

                //(dist_x, dist_y) : distance between current pixel in patch and cell's center
                const int dist_x = -half_cell_size + (int)cell_thread_x - half_cell_size * cell_x;

                const int dist_y_begin = -half_cell_size - half_cell_size * (int)threadIdx.y;
                for (int dist_y = dist_y_begin; dist_y < dist_y_begin + patch_size; ++dist_y)
                {
                    float2 vote = *(const float2*)grad_ptr;
                    uchar2 bin = *(const uchar2*)qangle_ptr;

                    grad_ptr += grad.step/sizeof(float);
                    qangle_ptr += qangle.step;

                    //(dist_center_x, dist_center_y) : distance between current pixel in patch and block's center
                    int dist_center_y = dist_y - half_cell_size * (1 - 2 * cell_y);
                    int dist_center_x = dist_x - half_cell_size * (1 - 2 * cell_x);

                    float gaussian = ::expf(-(dist_center_y * dist_center_y +
                                              dist_center_x * dist_center_x) * scale);

                    float interp_weight = ((float)cell_size - ::fabs(dist_y + 0.5f)) *
                                          ((float)cell_size - ::fabs(dist_x + 0.5f)) / (float)threads_block;

                    hist[bin.x * block_patch_size * nblocks] += gaussian * interp_weight * vote.x;
                    hist[bin.y * block_patch_size * nblocks] += gaussian * interp_weight * vote.y;
                }

                //reduction of the histograms
                volatile float* hist_ = hist;
                for (int bin_id = 0; bin_id < cnbins; ++bin_id, hist_ += block_patch_size * nblocks)
                {
                    if (cell_thread_x < patch_size/2) hist_[0] += hist_[patch_size/2];
                    if (cell_thread_x < patch_size/4 && (!((patch_size/4) < 3 && cell_thread_x == 0)))
                            hist_[0] += hist_[patch_size/4];
                    if (cell_thread_x == 0)
                        final_hist[((cell_x + block_x * cncells_block_x) * cncells_block_y + cell_y) * cnbins + bin_id]
                            = hist_[0] + hist_[1] + hist_[2];
                }
            }

            __syncthreads();

            float* block_hist = block_hists + (blockIdx.y * img_block_width +
                                               blockIdx.x * blockDim.z + block_x) *
                                              cblock_hist_size;

            //copying from final_hist to block_hist
            int tid;
            if(threads_cell < cnbins)
            {
                tid = (cell_y * cncells_block_y + cell_x) * cnbins + cell_thread_x;
            } else
            {
                tid = (cell_y * cncells_block_y + cell_x) * threads_cell + cell_thread_x;
            }
            if (tid < cblock_hist_size)
            {
                block_hist[tid] = final_hist[block_x * cblock_hist_size + tid];
                if(threads_cell < cnbins && cell_thread_x == (threads_cell-1))
                {
                    for(int i=1;i<=(cnbins - threads_cell);++i)
                    {
                        block_hist[tid + i] = final_hist[block_x * cblock_hist_size + tid + i];
                    }
                }
            }
        }

        //declaration of variables and invoke the kernel with the calculated number of blocks
        void compute_hists(int nbins,
                           int block_stride_x, int block_stride_y,
                           int height, int width,
                           const PtrStepSzf& grad, const PtrStepSzb& qangle,
                           float sigma,
                           float* block_hists,
                           int cell_size_x, int cell_size_y,
                           int ncells_block_x, int ncells_block_y,
                           const cudaStream_t& stream,
                           bool should_sync = true,
                           int omlp_sem_od = -1,
                           bool should_lock = true)
        {
            const int ncells_block = ncells_block_x * ncells_block_y;
            const int patch_side = cell_size_x / 4;
            const int patch_size = cell_size_x + (patch_side * 2);
            const int block_patch_size = ncells_block * patch_size;
            const int threads_cell = power_2up(patch_size);
            const int threads_block = ncells_block * threads_cell;
            const int half_cell_size = cell_size_x / 2;

            int img_block_width = (width - ncells_block_x * cell_size_x + block_stride_x) /
                                  block_stride_x;
            int img_block_height = (height - ncells_block_y * cell_size_y + block_stride_y) /
                                   block_stride_y;

            const int nblocks = max_nblocks(threads_cell, ncells_block);
            dim3 grid(divUp(img_block_width, nblocks), img_block_height);
            dim3 threads(threads_cell * ncells_block_x, ncells_block_y, nblocks);

            // Precompute gaussian spatial window parameter
            float scale = 1.f / (2.f * sigma * sigma);

            int hists_size = (nbins * ncells_block * patch_size * nblocks) * sizeof(float);
            int final_hists_size = (nbins * ncells_block * nblocks) * sizeof(float);
            int smem = hists_size + final_hists_size;
            struct control_page* cp = get_ctrl_page();

            /* =============
             * LOCK: compute hists
             */
            if (should_lock)
                lock_fzlp(omlp_sem_od);
            wait_forbidden_zone(omlp_sem_od, NODE_C);

            lt_t fz_start = litmus_clock();

            if (nblocks == 4)
                compute_hists_kernel_many_blocks<4><<<grid, threads, smem, stream>>>(img_block_width, grad, qangle, scale, block_hists, cell_size_x, patch_size, block_patch_size, threads_cell, threads_block, half_cell_size);
            else if (nblocks == 3)
                compute_hists_kernel_many_blocks<3><<<grid, threads, smem, stream>>>(img_block_width, grad, qangle, scale, block_hists, cell_size_x, patch_size, block_patch_size, threads_cell, threads_block, half_cell_size);
            else if (nblocks == 2)
                compute_hists_kernel_many_blocks<2><<<grid, threads, smem, stream>>>(img_block_width, grad, qangle, scale, block_hists, cell_size_x, patch_size, block_patch_size, threads_cell, threads_block, half_cell_size);
            else
                compute_hists_kernel_many_blocks<1><<<grid, threads, smem, stream>>>(img_block_width, grad, qangle, scale, block_hists, cell_size_x, patch_size, block_patch_size, threads_cell, threads_block, half_cell_size);

            cp->fz_progress = FZ_POST_GPU_LAUNCH;
            exit_np();

            cudaSafeCall( cudaGetLastError() );

            if (true || should_sync)
            {
                cudaSafeCall(cudaStreamSynchronize(stream));
            }
            exit_forbidden_zone(omlp_sem_od);

            lt_t fz_len = litmus_clock() - fz_start;

            fprintf(stdout, "[%d | %d] Computation %d took %llu microseconds.\n",
                    gettid(), getpid(), NODE_C, fz_len / 1000);

            if (should_lock)
                unlock_fzlp(omlp_sem_od);
            /*
            * UNLOCK: compute hists
            * ============= */
        }


        //-------------------------------------------------------------
        //  Normalization of histograms via L2Hys_norm
        //


        template<int size>
        __device__ float reduce_smem(float* smem, float val)
        {
            unsigned int tid = threadIdx.x;
            float sum = val;

            reduce<size>(smem, sum, tid, plus<float>());

            if (size == 32)
            {
            #if __CUDA_ARCH__ >= 300
                return shfl(sum, 0);
            #else
                return smem[0];
            #endif
            }
            else
            {
            #if __CUDA_ARCH__ >= 300
                if (threadIdx.x == 0)
                    smem[0] = sum;
            #endif

                __syncthreads();

                return smem[0];
            }
        }


        template <int nthreads, // Number of threads which process one block historgam
                  int nblocks> // Number of block hisograms processed by one GPU thread block
        __global__ void normalize_hists_kernel_many_blocks(const int block_hist_size,
                                                           const int img_block_width,
                                                           float* block_hists, float threshold)
        {
            if (blockIdx.x * blockDim.z + threadIdx.z >= img_block_width)
                return;

            float* hist = block_hists + (blockIdx.y * img_block_width +
                                         blockIdx.x * blockDim.z + threadIdx.z) *
                                        block_hist_size + threadIdx.x;

            __shared__ float sh_squares[nthreads * nblocks];
            float* squares = sh_squares + threadIdx.z * nthreads;

            float elem = 0.f;
            if (threadIdx.x < block_hist_size)
                elem = hist[0];

            __syncthreads(); // prevent race condition (redundant?)
            float sum = reduce_smem<nthreads>(squares, elem * elem);

            float scale = 1.0f / (::sqrtf(sum) + 0.1f * block_hist_size);
            elem = ::min(elem * scale, threshold);

            __syncthreads(); // prevent race condition
            sum = reduce_smem<nthreads>(squares, elem * elem);

            scale = 1.0f / (::sqrtf(sum) + 1e-3f);

            if (threadIdx.x < block_hist_size)
                hist[0] = elem * scale;
        }


        void normalize_hists(int nbins,
                             int block_stride_x, int block_stride_y,
                             int height, int width,
                             float* block_hists,
                             float threshold,
                             int cell_size_x, int cell_size_y,
                             int ncells_block_x, int ncells_block_y,
                             const cudaStream_t& stream,
                             bool should_sync = true,
                             int omlp_sem_od = -1,
                             bool should_lock = true)
        {
            const int nblocks = 1;

            int block_hist_size = nbins * ncells_block_x * ncells_block_y;
            int nthreads = power_2up(block_hist_size);
            dim3 threads(nthreads, 1, nblocks);

            int img_block_width = (width - ncells_block_x * cell_size_x + block_stride_x) / block_stride_x;
            int img_block_height = (height - ncells_block_y * cell_size_y + block_stride_y) / block_stride_y;
            dim3 grid(divUp(img_block_width, nblocks), img_block_height);
            struct control_page* cp = get_ctrl_page();

            /* =============
             * LOCK: normalize hists
             */
            if (should_lock)
                lock_fzlp(omlp_sem_od);
            wait_forbidden_zone(omlp_sem_od, NODE_D);

            lt_t fz_start = litmus_clock();

            if (nthreads == 32)
                normalize_hists_kernel_many_blocks<32, nblocks><<<grid, threads, 0, stream>>>(block_hist_size, img_block_width, block_hists, threshold);
            else if (nthreads == 64)
                normalize_hists_kernel_many_blocks<64, nblocks><<<grid, threads, 0, stream>>>(block_hist_size, img_block_width, block_hists, threshold);
            else if (nthreads == 128)
                normalize_hists_kernel_many_blocks<128, nblocks><<<grid, threads, 0, stream>>>(block_hist_size, img_block_width, block_hists, threshold);
            else if (nthreads == 256)
                normalize_hists_kernel_many_blocks<256, nblocks><<<grid, threads, 0, stream>>>(block_hist_size, img_block_width, block_hists, threshold);
            else if (nthreads == 512)
                normalize_hists_kernel_many_blocks<512, nblocks><<<grid, threads, 0, stream>>>(block_hist_size, img_block_width, block_hists, threshold);
            else
                CV_Error(cv::Error::StsBadArg, "normalize_hists: histogram's size is too big, try to decrease number of bins");

            cp->fz_progress = FZ_POST_GPU_LAUNCH;
            exit_np();

            cudaSafeCall( cudaGetLastError() );

            if (true || should_sync)
            {
                cudaSafeCall(cudaStreamSynchronize(stream));
            }
            exit_forbidden_zone(omlp_sem_od);

            lt_t fz_len = litmus_clock() - fz_start;

            fprintf(stdout, "[%d | %d] Computation %d took %llu microseconds.\n",
                    gettid(), getpid(), NODE_D, fz_len / 1000);

            if (should_lock)
                unlock_fzlp(omlp_sem_od);
            /*
            * UNLOCK: normalize hists
            * ============= */
        }

        template <int nthreads, // Number of threads per one histogram block
                  int nblocks>  // Number of histogram block processed by single GPU thread block
        __global__ void classify_hists_kernel_many_blocks(const int img_win_width, const int img_block_width,
                                                          const int win_block_stride_x, const int win_block_stride_y,
                                                          const float* block_hists, const float* coefs,
                                                          float free_coef, float threshold, unsigned char* labels)
        {
            const int win_x = threadIdx.z;
            if (blockIdx.x * blockDim.z + win_x >= img_win_width)
                return;

            const float* hist = block_hists + (blockIdx.y * win_block_stride_y * img_block_width +
                                               blockIdx.x * win_block_stride_x * blockDim.z + win_x) *
                                              cblock_hist_size;

            float product = 0.f;
            for (int i = threadIdx.x; i < cdescr_size; i += nthreads)
            {
                int offset_y = i / cdescr_width;
                int offset_x = i - offset_y * cdescr_width;
                product += coefs[i] * hist[offset_y * img_block_width * cblock_hist_size + offset_x];
            }

            __shared__ float products[nthreads * nblocks];

            const int tid = threadIdx.z * nthreads + threadIdx.x;

            reduce<nthreads>(products, product, tid, plus<float>());

            if (threadIdx.x == 0)
                labels[blockIdx.y * img_win_width + blockIdx.x * blockDim.z + win_x] = (product + free_coef >= threshold);
        }


        void classify_hists(int win_height, int win_width, int block_stride_y, int block_stride_x,
                            int win_stride_y, int win_stride_x, int height, int width, float* block_hists,
                            float* coefs, float free_coef, float threshold, int cell_size_x, int ncells_block_x, unsigned char* labels,
                            const cudaStream_t& stream,
                            int omlp_sem_od = -1,
                            bool should_lock = true)
        {
            const int nthreads = 256;
            const int nblocks = 1;

            int win_block_stride_x = win_stride_x / block_stride_x;
            int win_block_stride_y = win_stride_y / block_stride_y;
            int img_win_width = (width - win_width + win_stride_x) / win_stride_x;
            int img_win_height = (height - win_height + win_stride_y) / win_stride_y;

            dim3 threads(nthreads, 1, nblocks);
            dim3 grid(divUp(img_win_width, nblocks), img_win_height);

            int img_block_width = (width - ncells_block_x * cell_size_x + block_stride_x) / block_stride_x;
            struct control_page* cp = get_ctrl_page();

            /* =============
             * LOCK: classify hists
             */
            if (should_lock)
                lock_fzlp(omlp_sem_od);
            wait_forbidden_zone(omlp_sem_od, NODE_E);

            lt_t fz_start = litmus_clock();

            cudaSafeCall(cudaFuncSetCacheConfig(classify_hists_kernel_many_blocks<nthreads, nblocks>, cudaFuncCachePreferL1));

            classify_hists_kernel_many_blocks<nthreads, nblocks><<<grid,
                threads, 0, stream>>>(
                img_win_width, img_block_width, win_block_stride_x, win_block_stride_y,
                block_hists, coefs, free_coef, threshold, labels);

            cp->fz_progress = FZ_POST_GPU_LAUNCH;
            exit_np();

            cudaSafeCall( cudaGetLastError() );

            cudaSafeCall(cudaStreamSynchronize(stream));
            exit_forbidden_zone(omlp_sem_od);

            lt_t fz_len = litmus_clock() - fz_start;

            fprintf(stdout, "[%d | %d] Computation %d took %llu microseconds.\n",
                    gettid(), getpid(), NODE_E, fz_len / 1000);

            if (should_lock)
                unlock_fzlp(omlp_sem_od);
            /*
            * UNLOCK: classify hists
            * ============= */
        }

        //---------------------------------------------------------------------
        //  Linear SVM based classification
        //

       // return confidence values not just positive location
       template <int nthreads, // Number of threads per one histogram block
                 int nblocks>  // Number of histogram block processed by single GPU thread block
       __global__ void compute_confidence_hists_kernel_many_blocks(const int img_win_width, const int img_block_width,
                                                                                                           const int win_block_stride_x, const int win_block_stride_y,
                                                                                                           const float* block_hists, const float* coefs,
                                                                                                           float free_coef, float threshold, float* confidences)
       {
           const int win_x = threadIdx.z;
           if (blockIdx.x * blockDim.z + win_x >= img_win_width)
                   return;

           const float* hist = block_hists + (blockIdx.y * win_block_stride_y * img_block_width +
                                                                                blockIdx.x * win_block_stride_x * blockDim.z + win_x) *
                                                                               cblock_hist_size;

           float product = 0.f;
           for (int i = threadIdx.x; i < cdescr_size; i += nthreads)
           {
                   int offset_y = i / cdescr_width;
                   int offset_x = i - offset_y * cdescr_width;
                   product += coefs[i] * hist[offset_y * img_block_width * cblock_hist_size + offset_x];
           }

           __shared__ float products[nthreads * nblocks];

           const int tid = threadIdx.z * nthreads + threadIdx.x;

           reduce<nthreads>(products, product, tid, plus<float>());

           if (threadIdx.x == 0)
               confidences[blockIdx.y * img_win_width + blockIdx.x * blockDim.z + win_x] = product + free_coef;

       }

       void compute_confidence_hists(int win_height, int win_width, int block_stride_y, int block_stride_x,
                                               int win_stride_y, int win_stride_x, int height, int width, float* block_hists,
                                               float* coefs, float free_coef, float threshold, int cell_size_x, int ncells_block_x, float *confidences,
                                               const cudaStream_t& stream,
                                               int omlp_sem_od = -1,
                                               bool should_lock = true)
       {
           const int nthreads = 256;
           const int nblocks = 1;

           int win_block_stride_x = win_stride_x / block_stride_x;
           int win_block_stride_y = win_stride_y / block_stride_y;
           int img_win_width = (width - win_width + win_stride_x) / win_stride_x;
           int img_win_height = (height - win_height + win_stride_y) / win_stride_y;

           dim3 threads(nthreads, 1, nblocks);
           dim3 grid(divUp(img_win_width, nblocks), img_win_height);

           int img_block_width = (width - ncells_block_x * cell_size_x + block_stride_x) /
                                                       block_stride_x;
           struct control_page* cp = get_ctrl_page();

           /* =============
            * LOCK: classify hists
            */
           if (should_lock)
               lock_fzlp(omlp_sem_od);
           wait_forbidden_zone(omlp_sem_od, NODE_E);

           lt_t fz_start = litmus_clock();

           cudaSafeCall(cudaFuncSetCacheConfig(compute_confidence_hists_kernel_many_blocks<nthreads, nblocks>,
                                                                                   cudaFuncCachePreferL1));

           compute_confidence_hists_kernel_many_blocks<nthreads,
               nblocks><<<grid, threads, 0, stream>>>(
                   img_win_width, img_block_width, win_block_stride_x, win_block_stride_y,
                   block_hists, coefs, free_coef, threshold, confidences);

           cp->fz_progress = FZ_POST_GPU_LAUNCH;
           exit_np();

           cudaSafeCall(cudaStreamSynchronize(stream));
           exit_forbidden_zone(omlp_sem_od);

           lt_t fz_len = litmus_clock() - fz_start;

           fprintf(stdout, "[%d | %d] Computation %d took %llu microseconds.\n",
                   gettid(), getpid(), NODE_E, fz_len / 1000);

           if (should_lock)
               unlock_fzlp(omlp_sem_od);
           /*
           * UNLOCK: classify hists
           * ============= */
       }

        //----------------------------------------------------------------------------
        // Gradients computation


        template <int nthreads, int correct_gamma>
        __global__ void compute_gradients_8UC4_kernel(int height, int width, const PtrStepb img,
                                                      float angle_scale, PtrStepf grad, PtrStepb qangle)
        {
            const int x = blockIdx.x * blockDim.x + threadIdx.x;

            const uchar4* row = (const uchar4*)img.ptr(blockIdx.y);

            __shared__ float sh_row[(nthreads + 2) * 3];

            uchar4 val;
            if (x < width)
                val = row[x];
            else
                val = row[width - 2];

            sh_row[threadIdx.x + 1] = val.x;
            sh_row[threadIdx.x + 1 + (nthreads + 2)] = val.y;
            sh_row[threadIdx.x + 1 + 2 * (nthreads + 2)] = val.z;

            if (threadIdx.x == 0)
            {
                val = row[::max(x - 1, 1)];
                sh_row[0] = val.x;
                sh_row[(nthreads + 2)] = val.y;
                sh_row[2 * (nthreads + 2)] = val.z;
            }

            if (threadIdx.x == blockDim.x - 1)
            {
                val = row[::min(x + 1, width - 2)];
                sh_row[blockDim.x + 1] = val.x;
                sh_row[blockDim.x + 1 + (nthreads + 2)] = val.y;
                sh_row[blockDim.x + 1 + 2 * (nthreads + 2)] = val.z;
            }

            __syncthreads();
            if (x < width)
            {
                float3 a, b;

                b.x = sh_row[threadIdx.x + 2];
                b.y = sh_row[threadIdx.x + 2 + (nthreads + 2)];
                b.z = sh_row[threadIdx.x + 2 + 2 * (nthreads + 2)];
                a.x = sh_row[threadIdx.x];
                a.y = sh_row[threadIdx.x + (nthreads + 2)];
                a.z = sh_row[threadIdx.x + 2 * (nthreads + 2)];

                float3 dx;
                if (correct_gamma)
                    dx = make_float3(::sqrtf(b.x) - ::sqrtf(a.x), ::sqrtf(b.y) - ::sqrtf(a.y), ::sqrtf(b.z) - ::sqrtf(a.z));
                else
                    dx = make_float3(b.x - a.x, b.y - a.y, b.z - a.z);

                float3 dy = make_float3(0.f, 0.f, 0.f);

                if (blockIdx.y > 0 && blockIdx.y < height - 1)
                {
                    val = ((const uchar4*)img.ptr(blockIdx.y - 1))[x];
                    a = make_float3(val.x, val.y, val.z);

                    val = ((const uchar4*)img.ptr(blockIdx.y + 1))[x];
                    b = make_float3(val.x, val.y, val.z);

                    if (correct_gamma)
                        dy = make_float3(::sqrtf(b.x) - ::sqrtf(a.x), ::sqrtf(b.y) - ::sqrtf(a.y), ::sqrtf(b.z) - ::sqrtf(a.z));
                    else
                        dy = make_float3(b.x - a.x, b.y - a.y, b.z - a.z);
                }

                float best_dx = dx.x;
                float best_dy = dy.x;

                float mag0 = dx.x * dx.x + dy.x * dy.x;
                float mag1 = dx.y * dx.y + dy.y * dy.y;
                if (mag0 < mag1)
                {
                    best_dx = dx.y;
                    best_dy = dy.y;
                    mag0 = mag1;
                }

                mag1 = dx.z * dx.z + dy.z * dy.z;
                if (mag0 < mag1)
                {
                    best_dx = dx.z;
                    best_dy = dy.z;
                    mag0 = mag1;
                }

                mag0 = ::sqrtf(mag0);

                float ang = (::atan2f(best_dy, best_dx) + CV_PI_F) * angle_scale - 0.5f;
                int hidx = (int)::floorf(ang);
                ang -= hidx;
                hidx = (hidx + cnbins) % cnbins;

                ((uchar2*)qangle.ptr(blockIdx.y))[x] = make_uchar2(hidx, (hidx + 1) % cnbins);
                ((float2*)grad.ptr(blockIdx.y))[x] = make_float2(mag0 * (1.f - ang), mag0 * ang);
            }
        }


        void compute_gradients_8UC4(int nbins,
                                    int height, int width, const PtrStepSzb& img,
                                    float angle_scale,
                                    PtrStepSzf grad, PtrStepSzb qangle,
                                    bool correct_gamma,
                                    const cudaStream_t& stream,
                                    bool should_sync = true,
                                    int omlp_sem_od = -1,
                                    bool should_lock = true)
        {
            (void)nbins;
            const int nthreads = 256;

            dim3 bdim(nthreads, 1);
            dim3 gdim(divUp(width, bdim.x), divUp(height, bdim.y));
            struct control_page* cp = get_ctrl_page();

            /* =============
             * LOCK: compute gradients
             */
            if (should_lock)
                lock_fzlp(omlp_sem_od);
            wait_forbidden_zone(omlp_sem_od, NODE_B);

            lt_t fz_start = litmus_clock();

            if (correct_gamma)
                compute_gradients_8UC4_kernel<nthreads, 1><<<gdim, bdim, 0, stream>>>(height, width, img, angle_scale, grad, qangle);
            else
                compute_gradients_8UC4_kernel<nthreads, 0><<<gdim, bdim, 0, stream>>>(height, width, img, angle_scale, grad, qangle);

            cp->fz_progress = FZ_POST_GPU_LAUNCH;
            exit_np();

            cudaSafeCall( cudaGetLastError() );

            if (true || should_sync)
            {
                cudaSafeCall(cudaStreamSynchronize(stream));
            }
            exit_forbidden_zone(omlp_sem_od);

            lt_t fz_len = litmus_clock() - fz_start;

            fprintf(stdout, "[%d | %d] Computation %d took %llu microseconds.\n",
                    gettid(), getpid(), NODE_B, fz_len / 1000);

            if (should_lock)
                unlock_fzlp(omlp_sem_od);
            /*
                * UNLOCK: compute gradients
                * ============= */
        }

        template <int nthreads, int correct_gamma>
        __global__ void compute_gradients_8UC1_kernel(int height, int width, const PtrStepb img,
                                                      float angle_scale, PtrStepf grad, PtrStepb qangle)
        {
            const int x = blockIdx.x * blockDim.x + threadIdx.x;

            const unsigned char* row = (const unsigned char*)img.ptr(blockIdx.y);

            __shared__ float sh_row[nthreads + 2];

            if (x < width)
                sh_row[threadIdx.x + 1] = row[x];
            else
                sh_row[threadIdx.x + 1] = row[width - 2];

            if (threadIdx.x == 0)
                sh_row[0] = row[::max(x - 1, 1)];

            if (threadIdx.x == blockDim.x - 1)
                sh_row[blockDim.x + 1] = row[::min(x + 1, width - 2)];

            __syncthreads();
            if (x < width)
            {
                float dx;

                if (correct_gamma)
                    dx = ::sqrtf(sh_row[threadIdx.x + 2]) - ::sqrtf(sh_row[threadIdx.x]);
                else
                    dx = sh_row[threadIdx.x + 2] - sh_row[threadIdx.x];

                float dy = 0.f;
                if (blockIdx.y > 0 && blockIdx.y < height - 1)
                {
                    float a = ((const unsigned char*)img.ptr(blockIdx.y + 1))[x];
                    float b = ((const unsigned char*)img.ptr(blockIdx.y - 1))[x];
                    if (correct_gamma)
                        dy = ::sqrtf(a) - ::sqrtf(b);
                    else
                        dy = a - b;
                }
                float mag = ::sqrtf(dx * dx + dy * dy);

                float ang = (::atan2f(dy, dx) + CV_PI_F) * angle_scale - 0.5f;
                int hidx = (int)::floorf(ang);
                ang -= hidx;
                hidx = (hidx + cnbins) % cnbins;

                ((uchar2*)qangle.ptr(blockIdx.y))[x] = make_uchar2(hidx, (hidx + 1) % cnbins);
                ((float2*)  grad.ptr(blockIdx.y))[x] = make_float2(mag * (1.f - ang), mag * ang);
            }
        }


        void compute_gradients_8UC1(int nbins,
                                    int height, int width, const PtrStepSzb& img,
                                    float angle_scale,
                                    PtrStepSzf grad, PtrStepSzb qangle,
                                    bool correct_gamma,
                                    const cudaStream_t& stream,
                                    bool should_sync = true,
                                    int omlp_sem_od = -1,
                                    bool should_lock = true)
        {
            (void)nbins;
            const int nthreads = 256;

            dim3 bdim(nthreads, 1);
            dim3 gdim(divUp(width, bdim.x), divUp(height, bdim.y));
            struct control_page* cp = get_ctrl_page();

            /* =============
             * LOCK: compute gradients
             */
            if (should_lock)
                lock_fzlp(omlp_sem_od);
            wait_forbidden_zone(omlp_sem_od, NODE_B);

            lt_t fz_start = litmus_clock();

            if (correct_gamma)
                compute_gradients_8UC1_kernel<nthreads, 1><<<gdim, bdim, 0, stream>>>(height, width, img, angle_scale, grad, qangle);
            else
                compute_gradients_8UC1_kernel<nthreads, 0><<<gdim, bdim, 0, stream>>>(height, width, img, angle_scale, grad, qangle);

            cp->fz_progress = FZ_POST_GPU_LAUNCH;
            exit_np();

            cudaSafeCall( cudaGetLastError() );

            if (true || should_sync)
            {
                cudaSafeCall(cudaStreamSynchronize(stream));
            }
            exit_forbidden_zone(omlp_sem_od);

            lt_t fz_len = litmus_clock() - fz_start;

            fprintf(stdout, "[%d | %d] Computation %d took %llu microseconds.\n",
                    gettid(), getpid(), NODE_B, fz_len / 1000);

            if (should_lock)
                unlock_fzlp(omlp_sem_od);
            /*
             * UNLOCK: compute gradients
             * ============= */
        }



        //-------------------------------------------------------------------
        // Resize

#define TEX_NUM 8
        texture<uchar4, 2, cudaReadModeNormalizedFloat> resize8UC4_tex;
        texture<uchar,  2, cudaReadModeNormalizedFloat> resize8UC1_tex;
        texture<uchar, 2, cudaReadModeNormalizedFloat> resize8UC1_tex_local0;
        texture<uchar, 2, cudaReadModeNormalizedFloat> resize8UC1_tex_local1;
        texture<uchar, 2, cudaReadModeNormalizedFloat> resize8UC1_tex_local2;
        texture<uchar, 2, cudaReadModeNormalizedFloat> resize8UC1_tex_local3;
        texture<uchar, 2, cudaReadModeNormalizedFloat> resize8UC1_tex_local4;
        texture<uchar, 2, cudaReadModeNormalizedFloat> resize8UC1_tex_local5;
        texture<uchar, 2, cudaReadModeNormalizedFloat> resize8UC1_tex_local6;
        texture<uchar, 2, cudaReadModeNormalizedFloat> resize8UC1_tex_local7;
        texture<uchar4, 2, cudaReadModeNormalizedFloat> resize8UC4_tex_local0;
        texture<uchar4, 2, cudaReadModeNormalizedFloat> resize8UC4_tex_local1;
        texture<uchar4, 2, cudaReadModeNormalizedFloat> resize8UC4_tex_local2;
        texture<uchar4, 2, cudaReadModeNormalizedFloat> resize8UC4_tex_local3;
        texture<uchar4, 2, cudaReadModeNormalizedFloat> resize8UC4_tex_local4;
        texture<uchar4, 2, cudaReadModeNormalizedFloat> resize8UC4_tex_local5;
        texture<uchar4, 2, cudaReadModeNormalizedFloat> resize8UC4_tex_local6;
        texture<uchar4, 2, cudaReadModeNormalizedFloat> resize8UC4_tex_local7;

        __global__ void resize_for_hog_kernel(float sx, float sy,
                PtrStepSz<uchar> dst, int colOfs, int tex_index)
        {
            unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
            unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x < dst.cols && y < dst.rows)
                switch(tex_index)
                {
                    case 0:
                        dst.ptr(y)[x] = tex2D(resize8UC1_tex_local0, x * sx + colOfs, y * sy) * 255;
                        break;
                    case 1:
                        dst.ptr(y)[x] = tex2D(resize8UC1_tex_local1, x * sx + colOfs, y * sy) * 255;
                        break;
                    case 2:
                        dst.ptr(y)[x] = tex2D(resize8UC1_tex_local2, x * sx + colOfs, y * sy) * 255;
                        break;
                    case 3:
                        dst.ptr(y)[x] = tex2D(resize8UC1_tex_local3, x * sx + colOfs, y * sy) * 255;
                        break;
                    case 4:
                        dst.ptr(y)[x] = tex2D(resize8UC1_tex_local4, x * sx + colOfs, y * sy) * 255;
                        break;
                    case 5:
                        dst.ptr(y)[x] = tex2D(resize8UC1_tex_local5, x * sx + colOfs, y * sy) * 255;
                        break;
                    case 6:
                        dst.ptr(y)[x] = tex2D(resize8UC1_tex_local6, x * sx + colOfs, y * sy) * 255;
                        break;
                    case 7:
                        dst.ptr(y)[x] = tex2D(resize8UC1_tex_local7, x * sx + colOfs, y * sy) * 255;
                        break;
                    default:
                        return;
                }
        }

        __global__ void resize_for_hog_kernel(float sx, float sy,
                PtrStepSz<uchar4> dst, int colOfs, int tex_index)
        {
            unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
            unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x < dst.cols && y < dst.rows)
            {
                float4 val;
                switch(tex_index)
                {
                    case 0:
                        val = tex2D(resize8UC4_tex_local0, x * sx + colOfs, y * sy);
                        break;
                    case 1:
                        val = tex2D(resize8UC4_tex_local1, x * sx + colOfs, y * sy);
                        break;
                    case 2:
                        val = tex2D(resize8UC4_tex_local2, x * sx + colOfs, y * sy);
                        break;
                    case 3:
                        val = tex2D(resize8UC4_tex_local3, x * sx + colOfs, y * sy);
                        break;
                    case 4:
                        val = tex2D(resize8UC4_tex_local4, x * sx + colOfs, y * sy);
                        break;
                    case 5:
                        val = tex2D(resize8UC4_tex_local5, x * sx + colOfs, y * sy);
                        break;
                    case 6:
                        val = tex2D(resize8UC4_tex_local6, x * sx + colOfs, y * sy);
                        break;
                    case 7:
                        val = tex2D(resize8UC4_tex_local7, x * sx + colOfs, y * sy);
                        break;
                    default:
                        return;
                }
                dst.ptr(y)[x] = make_uchar4(val.x * 255, val.y * 255, val.z * 255, val.w * 255);
            }
        }

        template<class T, class TEX>
        static void resize_for_hog(const PtrStepSzb& src, PtrStepSzb dst, TEX& tex,
                                   const cudaStream_t& stream, int tex_index, bool should_sync, int omlp_sem_od,
                                   bool should_lock = true)
        {
            tex.filterMode = cudaFilterModeLinear;

            size_t texOfs = 0;
            int colOfs = 0;

            dim3 threads(32, 8);
            dim3 grid(divUp(dst.cols, threads.x), divUp(dst.rows, threads.y));

            float sx = static_cast<float>(src.cols) / dst.cols;
            float sy = static_cast<float>(src.rows) / dst.rows;
            struct control_page* cp = get_ctrl_page();

            /* =============
             * LOCK: resize
             */
            if (should_lock)
                lock_fzlp(omlp_sem_od);
            wait_forbidden_zone(omlp_sem_od, NODE_A);

            lt_t fz_start = litmus_clock();

            cudaChannelFormatDesc desc = cudaCreateChannelDesc<T>();
            cudaSafeCall( cudaBindTexture2D(&texOfs, tex, src.data, desc, src.cols, src.rows, src.step) );

            if (texOfs != 0)
            {
                colOfs = static_cast<int>( texOfs/sizeof(T) );
                cudaSafeCall( cudaUnbindTexture(tex) );
                cudaSafeCall( cudaBindTexture2D(&texOfs, tex, src.data, desc, src.cols, src.rows, src.step) );
            }

            resize_for_hog_kernel<<<grid, threads, 0, stream>>>(sx, sy, (PtrStepSz<T>)dst, colOfs, tex_index);

            cp->fz_progress = FZ_POST_GPU_LAUNCH;
            exit_np();

            cudaSafeCall( cudaGetLastError() );

            if (true || should_sync)
            {
                cudaSafeCall(cudaStreamSynchronize(stream));
            }
            exit_forbidden_zone(omlp_sem_od);

            lt_t fz_len = litmus_clock() - fz_start;

            fprintf(stdout, "[%d | %d] Computation %d took %llu microseconds.\n",
                    gettid(), getpid(), NODE_A, fz_len / 1000);

            if (should_lock)
                unlock_fzlp(omlp_sem_od);
            /*
             * UNLOCK: resize
             * ============= */
        }

        void resize_8UC1(const PtrStepSzb& src, PtrStepSzb dst,
                         const cudaStream_t& stream, bool should_sync = true, int omlp_sem_od = -1,
                         bool should_lock = true)
        {
            resize_for_hog<uchar> (src, dst, resize8UC1_tex_local0, stream, 0, should_sync, omlp_sem_od, should_lock);
        }
        void resize_8UC4(const PtrStepSzb& src, PtrStepSzb dst,
                         const cudaStream_t& stream, bool should_sync = true, int omlp_sem_od = -1,
                         bool should_lock = true)
        {
            resize_for_hog<uchar4>(src, dst, resize8UC4_tex_local0, stream, 0, should_sync, omlp_sem_od, should_lock);
        }

        void resize_8UC1_thread_safe(const PtrStepSzb& src, PtrStepSzb dst,
                                     const cudaStream_t& stream, int index,
                                     bool should_sync = true, int omlp_sem_od = -1,
                                     bool should_lock = true)
        {
            int tex_index = index % TEX_NUM;
            switch(tex_index)
            {
                case 0:
                    resize_for_hog<uchar>(src, dst, resize8UC1_tex_local0, stream, tex_index, should_sync, omlp_sem_od, should_lock);
                    break;
                case 1:
                    resize_for_hog<uchar>(src, dst, resize8UC1_tex_local1, stream, tex_index, should_sync, omlp_sem_od, should_lock);
                    break;
                case 2:
                    resize_for_hog<uchar>(src, dst, resize8UC1_tex_local2, stream, tex_index, should_sync, omlp_sem_od, should_lock);
                    break;
                case 3:
                    resize_for_hog<uchar>(src, dst, resize8UC1_tex_local3, stream, tex_index, should_sync, omlp_sem_od, should_lock);
                    break;
                case 4:
                    resize_for_hog<uchar>(src, dst, resize8UC1_tex_local4, stream, tex_index, should_sync, omlp_sem_od, should_lock);
                    break;
                case 5:
                    resize_for_hog<uchar>(src, dst, resize8UC1_tex_local5, stream, tex_index, should_sync, omlp_sem_od, should_lock);
                    break;
                case 6:
                    resize_for_hog<uchar>(src, dst, resize8UC1_tex_local6, stream, tex_index, should_sync, omlp_sem_od, should_lock);
                    break;
                case 7:
                    resize_for_hog<uchar>(src, dst, resize8UC1_tex_local7, stream, tex_index, should_sync, omlp_sem_od, should_lock);
                    break;
                default:
                    return;
            }
        }

        void resize_8UC4_thread_safe(const PtrStepSzb& src, PtrStepSzb dst,
                                     const cudaStream_t& stream, int index,
                                     bool should_sync = true, int omlp_sem_od = -1,
                                     bool should_lock = true)
        {
            int tex_index = index % TEX_NUM;
            switch(tex_index)
            {
                case 0:
                    resize_for_hog<uchar4>(src, dst, resize8UC4_tex_local0, stream, tex_index, should_sync, omlp_sem_od, should_lock);
                    break;
                case 1:
                    resize_for_hog<uchar4>(src, dst, resize8UC4_tex_local1, stream, tex_index, should_sync, omlp_sem_od, should_lock);
                    break;
                case 2:
                    resize_for_hog<uchar4>(src, dst, resize8UC4_tex_local2, stream, tex_index, should_sync, omlp_sem_od, should_lock);
                    break;
                case 3:
                    resize_for_hog<uchar4>(src, dst, resize8UC4_tex_local3, stream, tex_index, should_sync, omlp_sem_od, should_lock);
                    break;
                case 4:
                    resize_for_hog<uchar4>(src, dst, resize8UC4_tex_local4, stream, tex_index, should_sync, omlp_sem_od, should_lock);
                    break;
                case 5:
                    resize_for_hog<uchar4>(src, dst, resize8UC4_tex_local5, stream, tex_index, should_sync, omlp_sem_od, should_lock);
                    break;
                case 6:
                    resize_for_hog<uchar4>(src, dst, resize8UC4_tex_local6, stream, tex_index, should_sync, omlp_sem_od, should_lock);
                    break;
                case 7:
                    resize_for_hog<uchar4>(src, dst, resize8UC4_tex_local7, stream, tex_index, should_sync, omlp_sem_od, should_lock);
                    break;
                default:
                    return;
            }
        }
    } // namespace hog_rt
}}} // namespace cv { namespace cuda { namespace cudev


#endif /* CUDA_DISABLER */
