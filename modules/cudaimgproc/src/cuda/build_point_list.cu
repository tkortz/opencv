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
#include "opencv2/core/cuda/emulation.hpp"
#include "opencv2/cudaimgproc.hpp"

#include <unistd.h>

namespace cv { namespace cuda { namespace device
{
    namespace hough_locks
    {
        int use_locks = 0;

        /* resource_id must be a non-negative int */
        int open_fzlp_lock(int resource_id)
        {
            if (!use_locks) return -3;

            fprintf(stdout, "[%d | %d] Attempting to open OMLP (%d) semaphore now.\n", gettid(), getpid(), OMLP_SEM);

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
                else {
                    fprintf(stdout, "[%d | %d] Successfully opened OMLP semaphore lock: %d.\n", gettid(), getpid(), lock_od);
                }
            }

            return lock_od;
        }

        int lock_fzlp(int sem_od)
        {
            if (!use_locks) return -3;

            int res = -2;

            if (sem_od >= 0)
            {
                fprintf(stdout, "[%d | %d] Calling lock (%d) at time \t%llu\n", gettid(), getpid(), sem_od, litmus_clock());
                res = litmus_lock(sem_od);
                fprintf(stdout, "[%d | %d] Acquired lock at time \t%llu (status=%d)\n", gettid(), getpid(), litmus_clock(), res);
            }

            return res;
        }

        int wait_forbidden_zone(int sem_od, bool is_long = false)
        {
            if (!use_locks) return -3;

            int res = -2;

            int zone_length = 0;
            int cpu_measured = 0;

            // Just use defaults here
            if (is_long)
            {
                // HACK to support the d2d copy that frees and mallocs
                zone_length = ms2ns(10); // default to 10 ms
                cpu_measured = ms2ns(12); // default to 12 ms
            }
            else
            {
                zone_length = ms2ns(3); // default to 3 ms
                cpu_measured = ms2ns(4); // default to 4 ms
            }

            if (sem_od >= 0)
            {
                fprintf(stdout, "[%d | %d] Checking FZ at time \t%llu\n", gettid(), getpid(), litmus_clock());
                res = litmus_access_forbidden_zone_check(sem_od, cpu_measured, zone_length);
                fprintf(stdout, "[%d | %d] Not in FZ at time \t%llu (status=%d)\n", gettid(), getpid(), litmus_clock(), res);
            }

            return res;
        }

        int set_fz_launch_done(int sem_od)
        {
            if (!use_locks) return -3;
            int res = -2;
            if (sem_od >= 0)
                res = litmus_set_fz_launch_done(sem_od);
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
                fprintf(stdout, "[%d | %d] Unlocking at time \t\t%llu\n", gettid(), getpid(), litmus_clock());
                res = litmus_unlock(sem_od);
                fprintf(stdout, "[%d | %d] Unlocked at time \t\t%llu (status=%d)\n", gettid(), getpid(), litmus_clock(), res);
            }

            return res;
        }
    }

    namespace hough
    {
        __device__ int g_counter;

        template <int PIXELS_PER_THREAD>
        __global__ void buildPointList(const PtrStepSzb src, unsigned int* list)
        {
            __shared__ unsigned int s_queues[4][32 * PIXELS_PER_THREAD];
            __shared__ int s_qsize[4];
            __shared__ int s_globStart[4];

            const int x = blockIdx.x * blockDim.x * PIXELS_PER_THREAD + threadIdx.x;
            const int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (threadIdx.x == 0)
                s_qsize[threadIdx.y] = 0;
            __syncthreads();

            if (y < src.rows)
            {
                // fill the queue
                const uchar* srcRow = src.ptr(y);
                for (int i = 0, xx = x; i < PIXELS_PER_THREAD && xx < src.cols; ++i, xx += blockDim.x)
                {
                    if (srcRow[xx])
                    {
                        const unsigned int val = (y << 16) | xx;
                        const int qidx = Emulation::smem::atomicAdd(&s_qsize[threadIdx.y], 1);
                        s_queues[threadIdx.y][qidx] = val;
                    }
                }
            }

            __syncthreads();

            // let one thread reserve the space required in the global list
            if (threadIdx.x == 0 && threadIdx.y == 0)
            {
                // find how many items are stored in each list
                int totalSize = 0;
                for (int i = 0; i < blockDim.y; ++i)
                {
                    s_globStart[i] = totalSize;
                    totalSize += s_qsize[i];
                }

                // calculate the offset in the global list
                const int globalOffset = atomicAdd(&g_counter, totalSize);
                for (int i = 0; i < blockDim.y; ++i)
                    s_globStart[i] += globalOffset;
            }

            __syncthreads();

            // copy local queues to global queue
            const int qsize = s_qsize[threadIdx.y];
            int gidx = s_globStart[threadIdx.y] + threadIdx.x;
            for(int i = threadIdx.x; i < qsize; i += blockDim.x, gidx += blockDim.x)
                list[gidx] = s_queues[threadIdx.y][i];
        }

        int buildPointList_gpu(PtrStepSzb src, unsigned int* list,
                               const cudaStream_t& stream,
                               int omlp_sem_od = -1)
        {
            const int PIXELS_PER_THREAD = 16;

            void* counterPtr;
            cudaSafeCall( cudaGetSymbolAddress(&counterPtr, g_counter) );

            const dim3 block(32, 4);
            const dim3 grid(divUp(src.cols, block.x * PIXELS_PER_THREAD), divUp(src.rows, block.y));

            /* =============
             * LOCK: buildPointList_gpu
             */
            hough_locks::lock_fzlp(omlp_sem_od);

            // Memset of counterPtr
            hough_locks::wait_forbidden_zone(omlp_sem_od);
            cudaSafeCall( cudaMemsetAsync(counterPtr, 0, sizeof(int), stream) );
            hough_locks::set_fz_launch_done(omlp_sem_od);
            cudaSafeCall( cudaStreamSynchronize(stream) );
            hough_locks::exit_forbidden_zone(omlp_sem_od);

            cudaSafeCall( cudaFuncSetCacheConfig(buildPointList<PIXELS_PER_THREAD>, cudaFuncCachePreferShared) );

            // Kernel: buildPointList
            hough_locks::wait_forbidden_zone(omlp_sem_od);
            buildPointList<PIXELS_PER_THREAD><<<grid, block, 0, stream>>>(src, list);
            cudaSafeCall( cudaGetLastError() );
            hough_locks::set_fz_launch_done(omlp_sem_od);
            cudaSafeCall( cudaDeviceSynchronize() );
            hough_locks::exit_forbidden_zone(omlp_sem_od);

            int totalCount;
            // Memcpy d2h of counterPtr
            hough_locks::wait_forbidden_zone(omlp_sem_od);
            cudaSafeCall( cudaMemcpyAsync(&totalCount, counterPtr, sizeof(int), cudaMemcpyDeviceToHost, stream) );
            hough_locks::set_fz_launch_done(omlp_sem_od);
            cudaSafeCall( cudaStreamSynchronize(stream) );
            hough_locks::exit_forbidden_zone(omlp_sem_od);

            hough_locks::unlock_fzlp(omlp_sem_od);
            /*
            * UNLOCK: buildPointList_gpu
            * ============= */

            return totalCount;
        }
    }
}}}

#endif /* CUDA_DISABLER */
