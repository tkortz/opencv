/*
 * This implementation of a lane detector is based on the Python code
 * in the following tutorial:
 *    https://www.analyticsvidhya.com/blog/2020/05/tutorial-real-time-lane-detection-opencv/
 *
 * It also pulls some file-processing code from the HOG GPU sample.
 */

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

#include "opencv2/core.hpp"
#include "opencv2/core/cuda_stream_accessor.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/cudaimgproc.hpp"

#include <litmus.h>
#include <unistd.h>

using namespace std;
using namespace cv;

#define CALL( exp ) do { \
        int ret; \
        ret = exp; \
        if (ret != 0) \
        fprintf(stderr, "%s failed: %m\n", #exp);\
        else \
        fprintf(stderr, "%s ok.\n", #exp); \
} while (0)

bool help_showed = false;

class Args
{
public:
    Args();
    static Args readArgs(int argc, char** argv);

    string src;
    bool src_is_folder;
    bool src_is_video;
    bool src_is_camera;
    int camera_id;

    int count;

    // Real-time parameters
    int period;
    int deadline;
    int phase;

    bool display;
};

class App
{
public:
    App(const Args& s);
    void run();

    void readFrames();
    void readImage(Mat& dst);
    void houghlines(cv::cuda::HoughSegmentDetector *hough,
                    Mat& src, Mat& img, int frame_num,
                    cv::cuda::GpuMat* gpu_d_src,
                    cv::cuda::GpuMat* gpu_d_lines,
                    cv::cuda::Stream& stream,
                    int omlp_sem_od);

private:
    App operator=(App&);

    void makeRealTimeTask(cv::Ptr<cv::cuda::HoughSegmentDetector> detector,
                          int *sem_od);

    Args args;
    bool running;

    unsigned int img_read_count;

    std::vector<Mat> frames;

    bool use_gpu;

    void handleKey(char key);

    string execTime(double timeSec) const;
    string foundCount(int found) const;
};

static void printHelp()
{
    cout << "Lane detector sample.\n"
         << "\nUsage: lane_detector\n"
         << "  (<image>|--video <vide>|--camera <camera_id>) # frames source\n"
         << "  or"
         << "  (--folder <folder_path>) # load images from folder\n"
         << " --count <number of frames to process if sequence (repeating if necessary)>\n"
         << " --period <period in ms>\n"
         << " --deadline <relative deadline in ms>\n"
         << " --phase <task offset in ms>\n"
         << " [--display <true/false>] # whether to display the resulting frame\n";
    help_showed = true;
}

int main(int argc, char** argv)
{
    try
    {
        Args args;
        if (argc < 2)
        {
            printHelp();
            args.camera_id = 0;
            args.src_is_camera = true;
        }
        else
        {
            args = Args::readArgs(argc, argv);
            if (help_showed)
                return -1;
        }
        App app(args);
        app.readFrames();
        app.run();
    }
    catch (const Exception& e) { return cout << "error: "  << e.what() << endl, 1; }
    catch (const exception& e) { return cout << "error: "  << e.what() << endl, 1; }
    catch(...) { return cout << "unknown exception" << endl, 1; }
    return 0;
}

Args::Args()
{
    src_is_video = false;
    src_is_camera = false;
    src_is_folder = false;
    camera_id = 0;

    count = 1000;

    period = 25; // ms
    deadline = 25; // ms
    phase = 0; // ms

    display = false;
}

Args Args::readArgs(int argc, char** argv)
{
    Args args;
    for (int i = 1; i < argc; i++)
    {
        if (string(argv[i]) == "--help") printHelp();
        else if (string(argv[i]) == "--video") { args.src = argv[++i]; args.src_is_video = true; }
        else if (string(argv[i]) == "--camera") { args.camera_id = atoi(argv[++i]); args.src_is_camera = true; }
        else if (string(argv[i]) == "--folder") { args.src = argv[++i]; args.src_is_folder = true;}
        else if (string(argv[i]) == "--count") {
            int count = atoi(argv[++i]);
            if (count < 0)
                throw runtime_error((string("negative number of frames: ") + argv[i]));
            args.count = count;
        }
        else if (string(argv[i]) == "--rt_period") {
            int period = atoi(argv[++i]);
            if (period <= 0)
                throw runtime_error(string("non-positive period: ") + argv[i]);
            args.period = period;
        }
        else if (string(argv[i]) == "--rt_deadline") {
            int deadline = atoi(argv[++i]);
            if (deadline <= 0)
                throw runtime_error(string("non-positive relative deadline: ") + argv[i]);
            args.deadline = deadline;
        }
        else if (string(argv[i]) == "--rt_phase") {
            int phase = atoi(argv[++i]);
            if (phase < 0)
                throw runtime_error(string("negative phase: ") + argv[i]);
            args.phase = phase;
        }
        else if (string(argv[i]) == "--display") args.display = (string(argv[++i]) == "true");
        else if (args.src.empty()) args.src = argv[i];
        else throw runtime_error(string("unknown key: ") + argv[i]);
    }

    return args;
}

App::App(const Args& s)
{
    cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice());

    args = s;
    cout << "\nControls:\n"
         << "\tESC - exit\n"
         << "\tm - change mode GPU <-> CPU\n"
         << endl;

    use_gpu = true;
}

void App::makeRealTimeTask(cv::Ptr<cv::cuda::HoughSegmentDetector> detector, int *sem_od)
{
    struct rt_task param;
    init_rt_task_param(&param);
    param.exec_cost = ms2ns(args.deadline); // use the relative deadline as the exec cost
    param.period = ms2ns(args.period);
    param.relative_deadline = ms2ns(args.deadline);
    param.phase = ms2ns(args.phase);
    param.budget_policy = NO_ENFORCEMENT;
    param.cls = RT_CLASS_SOFT;
    param.priority = LITMUS_LOWEST_PRIORITY;
    param.cpu = 0;

    CALL( init_litmus() );
    CALL( set_rt_task_param(gettid(), &param) );
    CALL( task_mode(LITMUS_RT_TASK) );
    CALL( wait_for_ts_release() );

    fprintf(stdout, "[%d | %d] Calling litmus_open_lock for OMLP_SEM.\n", gettid(), getpid());
    *sem_od = detector->open_lock(18); // use the cluster ID as the resource ID
    fprintf(stdout, "[%d | %d] Got OMLP_SEM=%d.\n", gettid(), getpid(), *sem_od);
}

void App::readFrames()
{
    if (args.src_is_camera)
    {
        return;
    }

    frames.clear();
    img_read_count = 1;

    while (true)
    {
        Mat f;
        readImage(f);

        if (f.empty())
        {
            break;
        }

        frames.push_back(f);

        if (!args.src_is_folder)
        {
            break;
        }
    }
}

void App::readImage(Mat& dst)
{
    VideoCapture vc;
    vector<String> filenames;

    if (args.src_is_video)
    {
        vc.open(args.src.c_str());
        if (!vc.isOpened())
            throw runtime_error(string("can't open video file: " + args.src));
        vc >> dst;
    }
    else if (args.src_is_folder) {
        String folder = args.src;
        glob(folder, filenames);
        dst = imread(filenames[img_read_count++]); // 0 --> .gitignore
        if (!dst.data)
            cerr << "Problem loading image from folder!!!" << endl;
    }
    else if (args.src_is_camera)
    {
        vc.open(args.camera_id);
        if (!vc.isOpened())
        {
            stringstream msg;
            msg << "can't open camera: " << args.camera_id;
            throw runtime_error(msg.str());
        }
        vc >> dst;
    }
    else
    {
        dst = imread(args.src);
        if (dst.empty())
            throw runtime_error(string("can't open image file: " + args.src));
    }
}

void App::houghlines(cv::cuda::HoughSegmentDetector *hough,
                     Mat& src, Mat& img, int frame_num,
                     cv::cuda::GpuMat* gpu_d_src,
                     cv::cuda::GpuMat* gpu_d_lines,
                     cv::cuda::Stream& stream,
                     int omlp_sem_od)
{
    double timeSec = 0.0;

    Mat src_copy;
    src.copyTo(src_copy);

    Mat mask;
    cv::Canny(img, mask, 100, 200, 3);

    vector<Vec4i> lines;

    if (!use_gpu)
    {
        const int64 start = getTickCount();

        cv::HoughLinesP(mask, lines, 1, CV_PI / 180, 30, 60, 200);

        timeSec = (getTickCount() - start) / getTickFrequency();
    }
    else
    {
        /* =============
         * LOCK: upload gpu_d_src
         */
        hough->lock_fzlp(omlp_sem_od);
        hough->wait_forbidden_zone(omlp_sem_od);
        gpu_d_src->upload(mask, stream);
        hough->set_fz_launch_done(omlp_sem_od);
        cudaStreamSynchronize(cv::cuda::StreamAccessor::getStream(stream));
        hough->exit_forbidden_zone(omlp_sem_od);
        hough->unlock_fzlp(omlp_sem_od);
        /*
         * UNLOCK: upload gpu_d_src
         * ============= */

        const int64 start = getTickCount();

        hough->detect(*gpu_d_src, *gpu_d_lines, stream, omlp_sem_od);

        timeSec = (getTickCount() - start) / getTickFrequency();

        if (!gpu_d_lines->empty())
        {
            lines.resize(gpu_d_lines->cols);
            Mat h_lines(1, gpu_d_lines->cols, CV_32SC4, &lines[0]);

            /* =============
             * LOCK: download gpu_d_lines
             */
            hough->lock_fzlp(omlp_sem_od);
            hough->wait_forbidden_zone(omlp_sem_od);
            gpu_d_lines->download(h_lines, stream);
            hough->set_fz_launch_done(omlp_sem_od);
            cudaStreamSynchronize(cv::cuda::StreamAccessor::getStream(stream));
            hough->exit_forbidden_zone(omlp_sem_od);
            hough->unlock_fzlp(omlp_sem_od);
            /*
             * UNLOCK: download gpu_d_lines
             * ============= */
        }
    }

    // Display the detected lanes
    if (args.display)
    {
        for (size_t i = 0; i < lines.size(); ++i)
        {
            Vec4i l = lines[i];
            line(src_copy, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 3, LINE_AA);
        }

        if (!use_gpu)
            putText(src_copy, "Mode: CPU", Point(5, 20), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 255), 2);
        else
            putText(src_copy, "Mode: GPU", Point(5, 20), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 255), 2);
        putText(src_copy, "Frame: " + std::to_string(frame_num), Point(5, 60), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 255), 2);
        putText(src_copy, "Time: " + execTime(timeSec), Point(5, 100), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 255), 2);
        putText(src_copy, "Found: " + foundCount(lines.size()), Point(5, 140), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 255), 2);

        imshow("Detected Lanes", src_copy);
    }
}

void App::run()
{
    Mat frame, img, img_aux, img_thresholded;

    Ptr<cv::cuda::HoughSegmentDetector> hough = cv::cuda::createHoughSegmentDetector(1.0f, (float) (CV_PI / 180.0f), 30, 200);

    // Use user-defined streams rather than the NULL stream
    cv::cuda::Stream stream;

    // Additional pointers: gpu_d_lines
    cv::cuda::GpuMat* gpu_d_lines = new cv::cuda::GpuMat();
    cv::cuda::GpuMat* gpu_d_src = new cv::cuda::GpuMat();

    cv::cuda::Stream managed_stream;
    cv::cuda::BufferPool pool(managed_stream);

    Size sz(1280, 720);
    *gpu_d_src = pool.getBuffer(sz, 0);

    int omlp_sem_od = -1;
    makeRealTimeTask(hough, &omlp_sem_od);

    int count_frame = 0;
    running = true;
    while (count_frame < args.count && running)
    {
        // Retrieve the image
        if (args.src_is_camera)
        {
            readImage(frame);
            if (frame.empty())
            {
                running = false;
                break;
            }
        }
        else
        {
            frame = frames[count_frame % frames.size()];
        }

        // Change format of the image
        cv::cvtColor(frame, img_aux, COLOR_BGR2GRAY);
        img_aux.copyTo(img);

        // Mask out the region directly in front of the car
        int w = frame.cols;
        int h = frame.rows;
        Mat stencil = Mat::zeros(frame.rows, frame.cols, CV_8UC1);

        vector<cv::Point> polygon;
        polygon.push_back(Point(w / 480 * 50,  h));
        polygon.push_back(Point(w / 480 * 220, h / 270 * 200));
        polygon.push_back(Point(w / 480 * 360, h / 270 * 200));
        polygon.push_back(Point(            w, h));

        cv::fillConvexPoly(stencil, polygon, Scalar(255,0,0), CV_AA, 0);

        cv::bitwise_and(img, stencil, img);

        // Threshold
        cv::threshold(img, img_thresholded, 160, 255, cv::THRESH_BINARY);

        // Run houghlines
        houghlines(hough, frame, img_thresholded, count_frame % frames.size(),
                   gpu_d_src, gpu_d_lines, stream, omlp_sem_od);

        handleKey((char)waitKey(3));

        count_frame++;

        sleep_next_period();
    }

    CALL( task_mode(BACKGROUND_TASK) );

    // Free additional pointers: gpu_d_lines, gpu_d_src
    gpu_d_lines->release();
    gpu_d_src->release();
}

void App::handleKey(char key)
{
    switch (key)
    {
    case 27:
        running = false;
        break;
    case 'm':
    case 'M':
        use_gpu = !use_gpu;
        cout << "Switched to " << (use_gpu ? "CUDA" : "CPU") << " mode\n";
        break;
    }
}

inline string App::execTime(double timeSec) const
{
    stringstream ss;
    ss << timeSec * 1000 << " ms";
    return ss.str();
}

inline string App::foundCount(int found) const
{
    stringstream ss;
    ss << found;
    return ss.str();
}