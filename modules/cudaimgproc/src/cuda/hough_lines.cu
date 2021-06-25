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

#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#include "opencv2/core/cuda/common.hpp"
#include "opencv2/core/cuda/emulation.hpp"
#include "opencv2/core/cuda/dynamic_smem.hpp"
#include "opencv2/cudaimgproc.hpp"

#include <unistd.h>

namespace cv { namespace cuda { namespace device
{
    namespace hough_lines_locks
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

        int wait_forbidden_zone(int sem_od)
        {
            if (!use_locks) return -3;

            int res = -2;

            int zone_length = 0;
            int cpu_measured = 0;

            // Just use defaults here
            zone_length = ms2ns(1); // default to 2 ms
            cpu_measured = ms2ns(3); // default to 3 ms

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

    namespace hough_lines
    {
        __device__ int g_counter;

        ////////////////////////////////////////////////////////////////////////
        // linesAccum

        __global__ void linesAccumGlobal(const unsigned int* list, const int count, PtrStepi accum, const float irho, const float theta, const int numrho)
        {
            const int n = blockIdx.x;
            const float ang = n * theta;

            float sinVal;
            float cosVal;
            sincosf(ang, &sinVal, &cosVal);
            sinVal *= irho;
            cosVal *= irho;

            const int shift = (numrho - 1) / 2;

            int* accumRow = accum.ptr(n + 1);
            for (int i = threadIdx.x; i < count; i += blockDim.x)
            {
                const unsigned int val = list[i];

                const int x = (val & 0xFFFF);
                const int y = (val >> 16) & 0xFFFF;

                int r = __float2int_rn(x * cosVal + y * sinVal);
                r += shift;

                ::atomicAdd(accumRow + r + 1, 1);
            }
        }

        __global__ void linesAccumShared(const unsigned int* list, const int count, PtrStepi accum, const float irho, const float theta, const int numrho)
        {
            int* smem = DynamicSharedMem<int>();

            for (int i = threadIdx.x; i < numrho + 1; i += blockDim.x)
                smem[i] = 0;

            __syncthreads();

            const int n = blockIdx.x;
            const float ang = n * theta;

            float sinVal;
            float cosVal;
            sincosf(ang, &sinVal, &cosVal);
            sinVal *= irho;
            cosVal *= irho;

            const int shift = (numrho - 1) / 2;

            for (int i = threadIdx.x; i < count; i += blockDim.x)
            {
                const unsigned int val = list[i];

                const int x = (val & 0xFFFF);
                const int y = (val >> 16) & 0xFFFF;

                int r = __float2int_rn(x * cosVal + y * sinVal);
                r += shift;

                Emulation::smem::atomicAdd(&smem[r + 1], 1);
            }

            __syncthreads();

            int* accumRow = accum.ptr(n + 1);
            for (int i = threadIdx.x; i < numrho + 1; i += blockDim.x)
                accumRow[i] = smem[i];
        }

        void linesAccum_gpu(const unsigned int* list, int count, PtrStepSzi accum, float rho, float theta, size_t sharedMemPerBlock, bool has20,
                            const cudaStream_t& stream,
                            int omlp_sem_od = -1)
        {
            const dim3 block(has20 ? 1024 : 512);
            const dim3 grid(accum.rows - 2);

            size_t smemSize = (accum.cols - 1) * sizeof(int);

            /* =============
             * LOCK: linesAccum_gpu
             */
            hough_lines_locks::lock_fzlp(omlp_sem_od);

            // Kernel: houghLinesProbabilistic
            hough_lines_locks::wait_forbidden_zone(omlp_sem_od);
            if (smemSize < sharedMemPerBlock - 1000)
                linesAccumShared<<<grid, block, smemSize, stream>>>(list, count, accum, 1.0f / rho, theta, accum.cols - 2);
            else
                linesAccumGlobal<<<grid, block, 0, stream>>>(list, count, accum, 1.0f / rho, theta, accum.cols - 2);
            cudaSafeCall( cudaGetLastError() );
            hough_lines_locks::set_fz_launch_done(omlp_sem_od);
            cudaSafeCall( cudaDeviceSynchronize() );
            hough_lines_locks::exit_forbidden_zone(omlp_sem_od);

            hough_lines_locks::unlock_fzlp(omlp_sem_od);
            /*
            * UNLOCK: linesAccum_gpu
            * ============= */
        }

        ////////////////////////////////////////////////////////////////////////
        // linesGetResult

        __global__ void linesGetResult(const PtrStepSzi accum, float2* out, int* votes, const int maxSize, const float rho, const float theta, const int threshold, const int numrho)
        {
            const int r = blockIdx.x * blockDim.x + threadIdx.x;
            const int n = blockIdx.y * blockDim.y + threadIdx.y;

            if (r >= accum.cols - 2 || n >= accum.rows - 2)
                return;

            const int curVotes = accum(n + 1, r + 1);

            if (curVotes > threshold &&
                curVotes >  accum(n + 1, r) &&
                curVotes >= accum(n + 1, r + 2) &&
                curVotes >  accum(n, r + 1) &&
                curVotes >= accum(n + 2, r + 1))
            {
                const float radius = (r - (numrho - 1) * 0.5f) * rho;
                const float angle = n * theta;

                const int ind = ::atomicAdd(&g_counter, 1);
                if (ind < maxSize)
                {
                    out[ind] = make_float2(radius, angle);
                    votes[ind] = curVotes;
                }
            }
        }

        int linesGetResult_gpu(PtrStepSzi accum, float2* out, int* votes, int maxSize, float rho, float theta, int threshold, bool doSort)
        {
            void* counterPtr;
            cudaSafeCall( cudaGetSymbolAddress(&counterPtr, g_counter) );

            cudaSafeCall( cudaMemset(counterPtr, 0, sizeof(int)) );

            const dim3 block(32, 8);
            const dim3 grid(divUp(accum.cols - 2, block.x), divUp(accum.rows - 2, block.y));

            cudaSafeCall( cudaFuncSetCacheConfig(linesGetResult, cudaFuncCachePreferL1) );

            linesGetResult<<<grid, block>>>(accum, out, votes, maxSize, rho, theta, threshold, accum.cols - 2);
            cudaSafeCall( cudaGetLastError() );

            cudaSafeCall( cudaDeviceSynchronize() );

            int totalCount;
            cudaSafeCall( cudaMemcpy(&totalCount, counterPtr, sizeof(int), cudaMemcpyDeviceToHost) );

            totalCount = ::min(totalCount, maxSize);

            if (doSort && totalCount > 0)
            {
                thrust::device_ptr<float2> outPtr(out);
                thrust::device_ptr<int> votesPtr(votes);
                thrust::sort_by_key(votesPtr, votesPtr + totalCount, outPtr, thrust::greater<int>());
            }

            return totalCount;
        }
    }
}}}


#endif /* CUDA_DISABLER */
