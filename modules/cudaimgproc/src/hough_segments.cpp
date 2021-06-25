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

#include "precomp.hpp"

#include "opencv2/cudaimgproc.hpp"
#include <unistd.h>

using namespace cv;
using namespace cv::cuda;

#if !defined (HAVE_CUDA) || defined (CUDA_DISABLER)

Ptr<cuda::HoughSegmentDetector> cv::cuda::createHoughSegmentDetector(float, float, int, int, int) { throw_no_cuda(); return Ptr<HoughSegmentDetector>(); }

#else /* !defined (HAVE_CUDA) */

namespace cv { namespace cuda { namespace device
{
    namespace hough_locks
    {
        int open_fzlp_lock(int resource_id);
        int lock_fzlp(int sem_od);
        int wait_forbidden_zone(int sem_od, bool is_long = false);
        int set_fz_launch_done(int sem_od);
        int exit_forbidden_zone(int sem_od);
        int unlock_fzlp(int sem_od);
    }

    namespace hough
    {
        int buildPointList_gpu(PtrStepSzb src, unsigned int* list,
                               const cudaStream_t& stream,
                               int omlp_sem_od = -1);
    }

    namespace hough_lines
    {
        void linesAccum_gpu(const unsigned int* list, int count, PtrStepSzi accum, float rho, float theta, size_t sharedMemPerBlock, bool has20,
                            const cudaStream_t& stream,
                            int omlp_sem_od = -1);
    }

    namespace hough_segments
    {
        int houghLinesProbabilistic_gpu(PtrStepSzb mask, PtrStepSzi accum, int4* out, int maxSize, float rho, float theta, int lineGap, int lineLength,
                                        const cudaStream_t& stream,
                                        int omlp_sem_od = -1);
    }
}}}

namespace
{
    class HoughSegmentDetectorImpl : public HoughSegmentDetector
    {
    public:
        HoughSegmentDetectorImpl(float rho, float theta, int minLineLength, int maxLineGap, int maxLines) :
            rho_(rho), theta_(theta), minLineLength_(minLineLength), maxLineGap_(maxLineGap), maxLines_(maxLines)
        {
        }

        void detect(InputArray src, OutputArray lines, Stream& stream, int omlp_sem_od);

        void setRho(float rho) { rho_ = rho; }
        float getRho() const { return rho_; }

        void setTheta(float theta) { theta_ = theta; }
        float getTheta() const { return theta_; }

        void setMinLineLength(int minLineLength) { minLineLength_ = minLineLength; }
        int getMinLineLength() const { return minLineLength_; }

        void setMaxLineGap(int maxLineGap) { maxLineGap_ = maxLineGap; }
        int getMaxLineGap() const { return maxLineGap_; }

        void setMaxLines(int maxLines) { maxLines_ = maxLines; }
        int getMaxLines() const { return maxLines_; }

        int open_lock(int resource_id);
        int lock_fzlp(int sem_od);
        int wait_forbidden_zone(int sem_od);
        int set_fz_launch_done(int sem_od);
        int exit_forbidden_zone(int sem_od);
        int unlock_fzlp(int sem_od);

        void write(FileStorage& fs) const
        {
            writeFormat(fs);
            fs << "name" << "PHoughLinesDetector_CUDA"
            << "rho" << rho_
            << "theta" << theta_
            << "minLineLength" << minLineLength_
            << "maxLineGap" << maxLineGap_
            << "maxLines" << maxLines_;
        }

        void read(const FileNode& fn)
        {
            CV_Assert( String(fn["name"]) == "PHoughLinesDetector_CUDA" );
            rho_ = (float)fn["rho"];
            theta_ = (float)fn["theta"];
            minLineLength_ = (int)fn["minLineLength"];
            maxLineGap_ = (int)fn["maxLineGap"];
            maxLines_ = (int)fn["maxLines"];
        }

    private:
        float rho_;
        float theta_;
        int minLineLength_;
        int maxLineGap_;
        int maxLines_;

        GpuMat accum_;
        GpuMat list_;
        GpuMat result_;
    };

    int HoughSegmentDetectorImpl::open_lock(int resource_id)
    {
        return cv::cuda::device::hough_locks::open_fzlp_lock(resource_id);
    }

    int HoughSegmentDetectorImpl::lock_fzlp(int sem_od)
    {
        return cv::cuda::device::hough_locks::lock_fzlp(sem_od);
    }

    int HoughSegmentDetectorImpl::wait_forbidden_zone(int sem_od)
    {
        return cv::cuda::device::hough_locks::wait_forbidden_zone(sem_od);
    }
    int HoughSegmentDetectorImpl::set_fz_launch_done(int sem_od)
    {
        return cv::cuda::device::hough_locks::set_fz_launch_done(sem_od);
    }

    int HoughSegmentDetectorImpl::exit_forbidden_zone(int sem_od)
    {
        return cv::cuda::device::hough_locks::exit_forbidden_zone(sem_od);
    }

    int HoughSegmentDetectorImpl::unlock_fzlp(int sem_od)
    {
        return cv::cuda::device::hough_locks::unlock_fzlp(sem_od);
    }

    void HoughSegmentDetectorImpl::detect(InputArray _src, OutputArray lines,
                                          Stream& stream,
                                          int omlp_sem_od)
    {
        using namespace cv::cuda::device;
        using namespace cv::cuda::device::hough;
        using namespace cv::cuda::device::hough_lines;
        using namespace cv::cuda::device::hough_segments;

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.type() == CV_8UC1 );
        CV_Assert( src.cols < std::numeric_limits<unsigned short>::max() );
        CV_Assert( src.rows < std::numeric_limits<unsigned short>::max() );

        ensureSizeIsEnough(1, src.size().area(), CV_32SC1, list_);
        unsigned int* srcPoints = list_.ptr<unsigned int>();

        const int pointsCount = buildPointList_gpu(src, srcPoints, StreamAccessor::getStream(stream));
        if (pointsCount == 0)
        {
            lines.release();
            return;
        }

        const int numangle = cvRound(CV_PI / theta_);
        const int numrho = cvRound(((src.cols + src.rows) * 2 + 1) / rho_);
        CV_Assert( numangle > 0 && numrho > 0 );

        ensureSizeIsEnough(numangle + 2, numrho + 2, CV_32SC1, accum_);

        /* =============
         * LOCK: memset accum_
         */
        hough_locks::lock_fzlp(omlp_sem_od);
        hough_locks::wait_forbidden_zone(omlp_sem_od);
        accum_.setTo(Scalar::all(0), stream);
        hough_locks::set_fz_launch_done(omlp_sem_od);
        cudaStreamSynchronize(StreamAccessor::getStream(stream));
        hough_locks::exit_forbidden_zone(omlp_sem_od);
        hough_locks::unlock_fzlp(omlp_sem_od);
        /*
         * UNLOCK: memset accum_
         * ============= */

        DeviceInfo devInfo;
        linesAccum_gpu(srcPoints, pointsCount, accum_, rho_, theta_, devInfo.sharedMemPerBlock(), devInfo.supports(FEATURE_SET_COMPUTE_20), StreamAccessor::getStream(stream));

        ensureSizeIsEnough(1, maxLines_, CV_32SC4, result_);

        int linesCount = houghLinesProbabilistic_gpu(src, accum_, result_.ptr<int4>(), maxLines_, rho_, theta_, maxLineGap_, minLineLength_, StreamAccessor::getStream(stream));

        if (linesCount == 0)
        {
            lines.release();
            return;
        }

        result_.cols = linesCount;
        /* =============
         * LOCK: d2d copy (and thus free/alloc) of lines/result_
         */
        hough_locks::lock_fzlp(omlp_sem_od);
        // TODO: find a less heavy-handed solution
        hough_locks::wait_forbidden_zone(omlp_sem_od, true);
        result_.copyTo(lines, stream); // d2d copy, also has free/malloc usually
        hough_locks::set_fz_launch_done(omlp_sem_od);
        cudaStreamSynchronize(StreamAccessor::getStream(stream));
        hough_locks::exit_forbidden_zone(omlp_sem_od); // done with d2d memcpy
        hough_locks::unlock_fzlp(omlp_sem_od);
        /*
         * UNLOCK: d2d copy (and thus free/alloc) of lines/result_
         * ============= */
    }
}

Ptr<HoughSegmentDetector> cv::cuda::createHoughSegmentDetector(float rho, float theta, int minLineLength, int maxLineGap, int maxLines)
{
    return makePtr<HoughSegmentDetectorImpl>(rho, theta, minLineLength, maxLineGap, maxLines);
}

#endif /* !defined (HAVE_CUDA) */
