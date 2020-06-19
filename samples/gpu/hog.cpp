#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <iomanip>
#include <stdexcept>
#include <opencv2/core/utility.hpp>
#include "opencv2/cudaobjdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/imgproc.hpp"

#include <unistd.h>
#include <errno.h>
#include <pthread.h>
#include <thread>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <cuda_runtime.h>
#include <pgm.h>

/* First, we include standard headers.
 * Generally speaking, a LITMUS^RT real-time task can perform any
 * system call, etc., but no real-time guarantees can be made if a
 * system call blocks. To be on the safe side, only use I/O for debugging
 * purposes and from non-real-time sections.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
/* Second, we include the LITMUS^RT user space library header.
 * This header, part of liblitmus, provides the user space API of
 * LITMUS^RT.
 */
#include <litmus.h>

/* Include mlockall() to lock all pages */
#include <sys/mman.h>

using namespace std;
using namespace cv;

int hog_sample_errors;

__thread char hog_sample_errstr[80];

//#define LOG_DEBUG 1
#define NUM_SCALE_LEVELS 13
#define FAIR_LATENESS_PP(m, period, cost) (period - (float)m * cost / (m - 1))
#define CheckError(e) \
do { int __ret = (e); \
if(__ret < 0) { \
    hog_sample_errors++; \
    char* errstr = strerror_r(errno, hog_sample_errstr, sizeof(errstr)); \
    fprintf(stderr, "%lu: Error %d (%s (%d)) @ %s:%s:%d\n",  \
            pthread_self(), __ret, errstr, errno, __FILE__, __FUNCTION__, __LINE__); \
}}while(0)


/* Catch errors.
 */
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
    static Args read(int argc, char** argv);

    string src;
    bool src_is_folder;
    bool src_is_video;
    bool src_is_camera;
    int camera_id;

    bool svm_load;
    string svm;

    bool write_video;
    string dst_video;
    double dst_video_fps;

    bool make_gray;

    bool resize_src;
    int width, height;

    double scale;
    int nlevels;
    int gr_threshold;

    double hit_threshold;
    bool hit_threshold_auto;

    int win_width;
    int win_stride_width, win_stride_height;
    int block_width;
    int block_stride_width, block_stride_height;
    int cell_width;
    int nbins;

    bool gamma_corr;

    scheduling_option sched;
    int count;
    bool display;
    int num_fine_graphs;
    int cluster;
    int task_id;
    bool realtime;
    bool early;
};


struct params_compute  // a.k.a. compute scales node
{
    cv::cuda::GpuMat * gpu_img;
    std::vector<Rect> * found;
    Mat * img_to_show;
    size_t frame_index;
    int64 start_time;
    int64 end_time;
};

struct params_normalize
{
    std::vector<Rect> * found;
    Mat * img_to_show;
    cv::cuda::GpuMat * smaller_img_array;
    cv::cuda::GpuMat * block_hists_array;
    std::vector<double> * level_scale;
    std::vector<double> * confidences;
    size_t frame_index;
    int64 start_time;
    int64 end_time;
};

struct params_classify
{
    std::vector<Rect> * found;
    Mat * img_to_show;
    cv::cuda::GpuMat * smaller_img_array;
    cv::cuda::GpuMat * block_hists_array;
    std::vector<double> * level_scale;
    std::vector<double> * confidences;
    size_t frame_index;
    int64 start_time;
    int64 end_time;
};

struct params_collect_locations
{
    std::vector<Rect> * found;
    Mat * img_to_show;
    cv::cuda::GpuMat * smaller_img_array;
    cv::cuda::GpuMat * block_hists_array;
    std::vector<double> * level_scale;
    std::vector<double> * confidences;
    cv::cuda::GpuMat * labels_array;
    size_t frame_index;
    int64 start_time;
    int64 end_time;
};

/* fine-grained */
struct params_resize
{
    cv::cuda::GpuMat * gpu_img;
    std::vector<Rect> * found;
    Mat * img_to_show;
    cv::cuda::GpuMat * smaller_img;
    cv::cuda::GpuMat * labels;
    std::vector<double> * level_scale;
    std::vector<double> * confidences;
    int index;
    size_t frame_index;
    int64 start_time;
    int64 end_time;
};

struct params_compute_gradients
{
    cv::cuda::GpuMat * gpu_img;
    std::vector<Rect> * found;
    Mat * img_to_show;
    cv::cuda::GpuMat * smaller_img;
    cv::cuda::GpuMat * labels;
    std::vector<double> * level_scale;
    std::vector<double> * confidences;
    int index;
    size_t frame_index;
    int64 start_time;
    int64 end_time;
};

struct params_compute_histograms
{
    cv::cuda::GpuMat * gpu_img;
    std::vector<Rect> * found;
    Mat * img_to_show;
    cv::cuda::GpuMat * smaller_img;
    cv::cuda::GpuMat * labels;
    cv::cuda::GpuMat * grad;
    cv::cuda::GpuMat * qangle;
    std::vector<double> * level_scale;
    std::vector<double> * confidences;
    int index;
    size_t frame_index;
    int64 start_time;
    int64 end_time;
};

struct params_fine_normalize
{
    cv::cuda::GpuMat * gpu_img;
    std::vector<Rect> * found;
    Mat * img_to_show;
    cv::cuda::GpuMat * smaller_img;
    cv::cuda::GpuMat * labels;
    cv::cuda::GpuMat * block_hists;
    std::vector<double> * level_scale;
    std::vector<double> * confidences;
    int index;
    size_t frame_index;
    int64 start_time;
    int64 end_time;
};

struct params_fine_classify
{
    cv::cuda::GpuMat * gpu_img;
    std::vector<Rect> * found;
    Mat * img_to_show;
    cv::cuda::GpuMat * smaller_img;
    cv::cuda::GpuMat * labels;
    cv::cuda::GpuMat * block_hists;
    std::vector<double> * level_scale;
    std::vector<double> * confidences;
    int index;
    size_t frame_index;
    int64 start_time;
    int64 end_time;
};

struct params_fine_collect_locations
{
    cv::cuda::GpuMat * gpu_img;
    std::vector<Rect> * found;
    Mat * img_to_show;
    cv::cuda::GpuMat * smaller_img;
    cv::cuda::GpuMat * labels;
    std::vector<double> * level_scale;
    std::vector<double> * confidences;
    int index;
    size_t frame_index;
    int64 start_time;
    int64 end_time;
};

struct params_display
{
    std::vector<Rect> * found;
    Mat * img_to_show;
    size_t frame_index;
    int64 start_time;
    int64 end_time;
};


class App
{
public:
    App(const Args& s);
    void run();

    void sched_etoe_hog(cv::Ptr<cv::cuda::HOG> gpu_hog, cv::HOGDescriptor cpu_hog);
    void sched_etoe_hog_preload(cv::Ptr<cv::cuda::HOG> gpu_hog, cv::HOGDescriptor cpu_hog, Mat* frames);
    void sched_coarse_grained_unrolled_for_hog(cv::Ptr<cv::cuda::HOG> gpu_hog, cv::HOGDescriptor cpu_hog, Mat* frames);
    void sched_fine_grained_hog(cv::Ptr<cv::cuda::HOG> gpu_hog, cv::HOGDescriptor cpu_hog, Mat* frames);

    void sched_AB_hog(cv::Ptr<cv::cuda::HOG> gpu_hog, cv::HOGDescriptor cpu_hog, Mat* frames);
    void sched_BC_hog(cv::Ptr<cv::cuda::HOG> gpu_hog, cv::HOGDescriptor cpu_hog, Mat* frames);
    void sched_CD_hog(cv::Ptr<cv::cuda::HOG> gpu_hog, cv::HOGDescriptor cpu_hog, Mat* frames);
    void sched_DE_hog(cv::Ptr<cv::cuda::HOG> gpu_hog, cv::HOGDescriptor cpu_hog, Mat* frames);

    void sched_ABC_hog(cv::Ptr<cv::cuda::HOG> gpu_hog, cv::HOGDescriptor cpu_hog, Mat* frames);
    void sched_BCD_hog(cv::Ptr<cv::cuda::HOG> gpu_hog, cv::HOGDescriptor cpu_hog, Mat* frames);
    void sched_CDE_hog(cv::Ptr<cv::cuda::HOG> gpu_hog, cv::HOGDescriptor cpu_hog, Mat* frames);

    void sched_ABCD_hog(cv::Ptr<cv::cuda::HOG> gpu_hog, cv::HOGDescriptor cpu_hog, Mat* frames);
    void sched_BCDE_hog(cv::Ptr<cv::cuda::HOG> gpu_hog, cv::HOGDescriptor cpu_hog, Mat* frames);

    void sched_ABCDE_hog(cv::Ptr<cv::cuda::HOG> gpu_hog, cv::HOGDescriptor cpu_hog, Mat* frames);

    void sched_single_merge_in_level_hog(cv::Ptr<cv::cuda::HOG> gpu_hog, cv::HOGDescriptor cpu_hog, Mat* frames);

    void thread_color_convert(node_t *_node, pthread_barrier_t* init_barrier,
            cv::Ptr<cv::cuda::HOG> gpu_hog, cv::HOGDescriptor cpu_hog, Mat* frames, struct task_info t_info, int graph_idx);

    void* thread_display(node_t* node, pthread_barrier_t* init_barrier, bool shouldDisplay);

    void handleKey(char key);

    void hogWorkBegin();
    void hogWorkEnd();
    string hogWorkFps() const;

    string frameIndex(int) const;

    void workBegin();
    void workEnd();
    string workFps() const;

    string message() const;

private:
    App operator=(App&);

    Args args;
    bool running;

    bool use_gpu;
    bool make_gray;
    double scale;
    int gr_threshold;
    int nlevels;
    double hit_threshold;
    bool gamma_corr;

    int64 hog_work_begin;
    int64 hog_work_end;
    double hog_work_fps;

    int64 work_begin;
    double work_fps;

    cv::VideoWriter video_writer;
};

static void printHelp()
{
    cout << "Histogram of Oriented Gradients descriptor and detector sample.\n"
         << "\nUsage: hog\n"
         << "  (<image>|--video <vide>|--camera <camera_id>) # frames source\n"
         << "  or"
         << "  (--folder <folder_path>) # load images from folder\n"
         << "  [--svm <file> # load svm file"
         << "  [--make_gray <true/false>] # convert image to gray one or not\n"
         << "  [--resize_src <true/false>] # do resize of the source image or not\n"
         << "  [--width <int>] # resized image width\n"
         << "  [--height <int>] # resized image height\n"
         << "  [--hit_threshold <double>] # classifying plane distance threshold (0.0 usually)\n"
         << "  [--scale <double>] # HOG window scale factor\n"
         << "  [--nlevels <int>] # max number of HOG window scales\n"
         << "  [--win_width <int>] # width of the window\n"
         << "  [--win_stride_width <int>] # distance by OX axis between neighbour wins\n"
         << "  [--win_stride_height <int>] # distance by OY axis between neighbour wins\n"
         << "  [--block_width <int>] # width of the block\n"
         << "  [--block_stride_width <int>] # distance by 0X axis between neighbour blocks\n"
         << "  [--block_stride_height <int>] # distance by 0Y axis between neighbour blocks\n"
         << "  [--cell_width <int>] # width of the cell\n"
         << "  [--nbins <int>] # number of bins\n"
         << "  [--gr_threshold <int>] # merging similar rects constant\n"
         << "  [--gamma_correct <int>] # do gamma correction or not\n"
         << "  [--write_video <bool>] # write video or not\n"
         << "  [--dst_video <path>] # output video path\n"
         << "  [--dst_video_fps <double>] # output video fps\n"
         << "  [--sched <int>] # scheduling option (0:end_to_end, 1:coarse_grained, 2:fine_grained, 3:unrolled_coarse_grained)\n"
         << "  [--count <int>] # num of frames to process\n"
         << "  [--graph_bound <int>] # response time bound of fine-grained HOG\n"
         << "  [--cluster <int>] # cluster ID of this task\n"
         << "  [--id <int>] # task ID of this task\n"
         << "  [--rt <true/false>] # run under LITMUS^RT scheduler or not\n"
         << "  [--display <true/false>] # to display result frame or not\n";

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
            args = Args::read(argc, argv);
            if (help_showed)
                return -1;
        }
        App app(args);
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
    svm_load = false;
    camera_id = 0;

    write_video = false;
    dst_video_fps = 24.;

    make_gray = false;

    resize_src = false;
    width = 640;
    height = 480;

    scale = 1.05;
    nlevels = 13;
    gr_threshold = 8;
    hit_threshold = 1.4;
    hit_threshold_auto = true;

    win_width = 48;
    win_stride_width = 8;
    win_stride_height = 8;
    block_width = 16;
    block_stride_width = 8;
    block_stride_height = 8;
    cell_width = 8;
    nbins = 9;

    gamma_corr = true;

    sched = fine_grained;
    count = 1000;
    num_fine_graphs = 1;
    display = false;
    cluster = -1;
    task_id = 0;
    realtime = true;
    early = true;
}


Args Args::read(int argc, char** argv)
{
    Args args;
    for (int i = 1; i < argc; i++)
    {
        if (string(argv[i]) == "--make_gray") args.make_gray = (string(argv[++i]) == "true");
        else if (string(argv[i]) == "--resize_src") args.resize_src = (string(argv[++i]) == "true");
        else if (string(argv[i]) == "--width") args.width = atoi(argv[++i]);
        else if (string(argv[i]) == "--height") args.height = atoi(argv[++i]);
        else if (string(argv[i]) == "--hit_threshold")
        {
            args.hit_threshold = atof(argv[++i]);
            args.hit_threshold_auto = false;
        }
        else if (string(argv[i]) == "--scale") args.scale = atof(argv[++i]);
        else if (string(argv[i]) == "--nlevels") args.nlevels = atoi(argv[++i]);
        else if (string(argv[i]) == "--win_width") args.win_width = atoi(argv[++i]);
        else if (string(argv[i]) == "--win_stride_width") args.win_stride_width = atoi(argv[++i]);
        else if (string(argv[i]) == "--win_stride_height") args.win_stride_height = atoi(argv[++i]);
        else if (string(argv[i]) == "--block_width") args.block_width = atoi(argv[++i]);
        else if (string(argv[i]) == "--block_stride_width") args.block_stride_width = atoi(argv[++i]);
        else if (string(argv[i]) == "--block_stride_height") args.block_stride_height = atoi(argv[++i]);
        else if (string(argv[i]) == "--cell_width") args.cell_width = atoi(argv[++i]);
        else if (string(argv[i]) == "--nbins") args.nbins = atoi(argv[++i]);
        else if (string(argv[i]) == "--gr_threshold") args.gr_threshold = atoi(argv[++i]);
        else if (string(argv[i]) == "--gamma_correct") args.gamma_corr = (string(argv[++i]) == "true");
        else if (string(argv[i]) == "--write_video") args.write_video = (string(argv[++i]) == "true");
        else if (string(argv[i]) == "--dst_video") args.dst_video = argv[++i];
        else if (string(argv[i]) == "--dst_video_fps") args.dst_video_fps = atof(argv[++i]);
        else if (string(argv[i]) == "--help") printHelp();
        else if (string(argv[i]) == "--video") { args.src = argv[++i]; args.src_is_video = true; }
        else if (string(argv[i]) == "--camera") { args.camera_id = atoi(argv[++i]); args.src_is_camera = true; }
        else if (string(argv[i]) == "--folder") { args.src = argv[++i]; args.src_is_folder = true;}
        else if (string(argv[i]) == "--svm") { args.svm = argv[++i]; args.svm_load = true;}
        else if (string(argv[i]) == "--sched") {
            int sched = atoi(argv[++i]);
            if (sched < 0 || sched >= scheduling_option_end)
                throw runtime_error((string("unknown scheduling option: ") + argv[i]));
            args.sched = (enum scheduling_option)sched;
        }
        else if (string(argv[i]) == "--count") {
            int count = atoi(argv[++i]);
            if (count < 0)
                throw runtime_error((string("negative number of frames: ") + argv[i]));
            args.count = count;
        }
        else if (string(argv[i]) == "--graph_bound") {
            int bound = atoi(argv[++i]);
            args.num_fine_graphs = (bound - 1) / PERIOD + 1; // floor
        }
        else if (string(argv[i]) == "--cluster") { args.cluster = atoi(argv[++i]); }
        else if (string(argv[i]) == "--id") { args.task_id = atoi(argv[++i]); }
        else if (string(argv[i]) == "--rt") args.realtime = (string(argv[++i]) == "true");
        else if (string(argv[i]) == "--early") args.early = (string(argv[++i]) == "true");
        else if (string(argv[i]) == "--display") args.display = (string(argv[++i]) == "true");
        else if (args.src.empty()) args.src = argv[i];
        else throw runtime_error((string("unknown key: ") + argv[i]));
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
         << "\tg - convert image to gray or not\n"
         << "\t1/q - increase/decrease HOG scale\n"
         << "\t2/w - increase/decrease levels count\n"
         << "\t3/e - increase/decrease HOG group threshold\n"
         << "\t4/r - increase/decrease hit threshold\n"
         << endl;

    use_gpu = true;
    make_gray = args.make_gray;
    scale = args.scale;
    gr_threshold = args.gr_threshold;
    nlevels = args.nlevels;

    if (args.hit_threshold_auto)
        args.hit_threshold = args.win_width == 48 ? 1.4 : 0.;
    hit_threshold = args.hit_threshold;

    gamma_corr = args.gamma_corr;

    cout << "Scale: " << scale << endl;
    if (args.resize_src)
        cout << "Resized source: (" << args.width << ", " << args.height << ")\n";
    cout << "Group threshold: " << gr_threshold << endl;
    cout << "Levels number: " << nlevels << endl;
    cout << "Win size: (" << args.win_width << ", " << args.win_width*2 << ")\n";
    cout << "Win stride: (" << args.win_stride_width << ", " << args.win_stride_height << ")\n";
    cout << "Block size: (" << args.block_width << ", " << args.block_width << ")\n";
    cout << "Block stride: (" << args.block_stride_width << ", " << args.block_stride_height << ")\n";
    cout << "Cell size: (" << args.cell_width << ", " << args.cell_width << ")\n";
    cout << "Bins number: " << args.nbins << endl;
    cout << "Hit threshold: " << hit_threshold << endl;
    cout << "Gamma correction: " << gamma_corr << endl;
    cout << endl;
}

struct linked_frames
{
    struct params_display * ptr;
    struct linked_frames * next;
};

void* App::thread_display(node_t* _node, pthread_barrier_t* init_barrier, bool shouldDisplay)
{
    node_t node = *_node;
#ifdef LOG_DEBUG
    char tabbuf[] = "\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t";
    tabbuf[node.node] = '\0';
#endif
    CheckError(pgm_claim_node(node));
    int ret = 0;

    edge_t *in_edge = (edge_t *)calloc(1, sizeof(edge_t));
    CheckError(pgm_get_edges_in(node, in_edge, 1));
    struct params_display *in_buf = (struct params_display *)pgm_get_edge_buf_c(*in_edge);
    if (in_buf == NULL)
        fprintf(stderr, "compute gradients in buffer is NULL\n");

    pthread_barrier_wait(init_barrier);

    if(!hog_sample_errors)
    {
        do {
            ret = pgm_wait(node);
            hogWorkEnd();

            if(ret != PGM_TERMINATE)
            {
                CheckError(ret);
#ifdef LOG_DEBUG
                fprintf(stdout, "%s%d fires (top)\n", tabbuf, node.node);
#endif


                printf("%lu response time: %f\n", in_buf->frame_index, (hog_work_end - in_buf->start_time) / getTickFrequency());

                // Draw positive classified windows
                if (shouldDisplay) {
                    for (size_t i = 0; i < in_buf->found->size(); i++) {
                        Rect r = (*in_buf->found)[i];
                        rectangle(*in_buf->img_to_show, r.tl(), r.br(), Scalar(0, 255, 0), 3);
//#ifdef LOG_DEBUG
                        fprintf(stdout, "point: %d, %d, %d, %d\n", r.tl().x, r.tl().y, r.br().x, r.br().y);
//#endif
                    }

                    if (use_gpu)
                        putText(*in_buf->img_to_show, "Mode: GPU", Point(5, 25), FONT_HERSHEY_SIMPLEX, 1., Scalar(255, 100, 0), 2);
                    else
                        putText(*in_buf->img_to_show, "Mode: CPU", Point(5, 25), FONT_HERSHEY_SIMPLEX, 1., Scalar(255, 100, 0), 2);
                    putText(*in_buf->img_to_show, "FPS HOG: " + hogWorkFps(), Point(5, 65), FONT_HERSHEY_SIMPLEX, 1., Scalar(255, 100, 0), 2);
                    putText(*in_buf->img_to_show, "FPS total: " + workFps(), Point(5, 105), FONT_HERSHEY_SIMPLEX, 1., Scalar(255, 100, 0), 2);
                    putText(*in_buf->img_to_show, "Frame: " + frameIndex(in_buf->frame_index), Point(5, 145), FONT_HERSHEY_SIMPLEX, 1., Scalar(255, 100, 0), 2);
                    imshow("opencv_gpu_hog", *in_buf->img_to_show);
                }

                workEnd();

                Mat * img = new Mat();
                if (args.write_video)
                {
                    if (!video_writer.isOpened())
                    {
                        video_writer.open(args.dst_video, VideoWriter::fourcc('x','v','i','d'), args.dst_video_fps,
                                in_buf->img_to_show->size(), true);
                        if (!video_writer.isOpened())
                            throw std::runtime_error("can't create video writer");
                    }

                    if (make_gray) cvtColor(*in_buf->img_to_show, *img, COLOR_GRAY2BGR);
                    else cvtColor(*in_buf->img_to_show, *img, COLOR_BGRA2BGR);

                    video_writer << *img;
                }

                handleKey((char)waitKey(3));
                delete in_buf->found;
                delete img;
                delete in_buf->img_to_show;

                //CheckError(pgm_complete(node));
            }
            else
            {
#ifdef LOG_DEBUG
                fprintf(stdout, "%s- %d terminates\n", tabbuf, node.node);
#endif
                //pgm_terminate(node);
            }

        } while(ret != PGM_TERMINATE);
    }

    pthread_barrier_wait(init_barrier);

    CheckError(pgm_release_node(node));

    free(in_edge);
    pthread_exit(0);
}

void App::sched_coarse_grained_unrolled_for_hog(cv::Ptr<cv::cuda::HOG> gpu_hog, cv::HOGDescriptor cpu_hog, Mat* frames)
{
    fprintf(stdout, "node name: color_convert(source), task id: 0, node tid: %d\n", gettid());
    /* graph construction */
    graph_t g;
    node_t color_convert_node;
    node_t compute_scales_node;
    node_t unrolled_vxHOGCells_node[NUM_SCALE_LEVELS];
    node_t normalize_node[NUM_SCALE_LEVELS];
    node_t classify_node[NUM_SCALE_LEVELS];
    node_t collect_locations_node;
    node_t display_node;
    edge_t e0_1;
    edge_t e1_2[NUM_SCALE_LEVELS];
    edge_t e2_3[NUM_SCALE_LEVELS];
    edge_t e3_4[NUM_SCALE_LEVELS];
    edge_t e4_5[NUM_SCALE_LEVELS];
    edge_t e5_6;

    char buf[30];
    sprintf(buf, "/tmp/graph_t%d", args.task_id);
    CheckError(pgm_init(buf, 1));
    CheckError(pgm_init_graph(&g, "hog"));

    CheckError(pgm_init_node(&color_convert_node, g, "color_convert"));
    CheckError(pgm_init_node(&compute_scales_node, g, "compute_scales"));
    for (int i=0; i<NUM_SCALE_LEVELS; i++) {
        CheckError(pgm_init_node(&unrolled_vxHOGCells_node[i], g, "unrolled_vxHOGCells"));
        CheckError(pgm_init_node(&normalize_node[i], g, "normalize"));
        CheckError(pgm_init_node(&classify_node[i], g, "classify"));
    }
    CheckError(pgm_init_node(&collect_locations_node, g, "collect_locations"));

    CheckError(pgm_init_node(&display_node, g, "display"));

    edge_attr_t fast_mq_attr;
    memset(&fast_mq_attr, 0, sizeof(fast_mq_attr));
    //fast_mq_attr.mq_maxmsg = 1; /* root required for higher values */
    //fast_mq_attr.type = pgm_fast_mq_edge;
    fast_mq_attr.type = pgm_fast_fifo_edge;

    fast_mq_attr.nr_produce = sizeof(struct params_compute);
    fast_mq_attr.nr_consume = sizeof(struct params_compute);
    fast_mq_attr.nr_threshold = sizeof(struct params_compute);
    CheckError(pgm_init_edge(&e0_1, color_convert_node, compute_scales_node, "e0_1", &fast_mq_attr));

    for (int i=0; i<NUM_SCALE_LEVELS; i++) {
        sprintf(buf, "e1_2_%d", i);
        fast_mq_attr.nr_produce = sizeof(struct params_resize);
        fast_mq_attr.nr_consume = sizeof(struct params_resize);
        fast_mq_attr.nr_threshold = sizeof(struct params_resize);
        CheckError(pgm_init_edge(&e1_2[i], compute_scales_node, unrolled_vxHOGCells_node[i], buf, &fast_mq_attr));

        sprintf(buf, "e2_3_%d", i);
        fast_mq_attr.nr_produce = sizeof(struct params_fine_normalize);
        fast_mq_attr.nr_consume = sizeof(struct params_fine_normalize);
        fast_mq_attr.nr_threshold = sizeof(struct params_fine_normalize);
        CheckError(pgm_init_edge(&e2_3[i], unrolled_vxHOGCells_node[i], normalize_node[i], buf, &fast_mq_attr));

        sprintf(buf, "e3_4_%d", i);
        fast_mq_attr.nr_produce = sizeof(struct params_fine_classify);
        fast_mq_attr.nr_consume = sizeof(struct params_fine_classify);
        fast_mq_attr.nr_threshold = sizeof(struct params_fine_classify);
        CheckError(pgm_init_edge(&e3_4[i], normalize_node[i], classify_node[i], buf, &fast_mq_attr));

        sprintf(buf, "e4_5_%d", i);
        fast_mq_attr.nr_produce = sizeof(struct params_fine_collect_locations);
        fast_mq_attr.nr_consume = sizeof(struct params_fine_collect_locations);
        fast_mq_attr.nr_threshold = sizeof(struct params_fine_collect_locations);
        CheckError(pgm_init_edge(&e4_5[i], classify_node[i], collect_locations_node, buf, &fast_mq_attr));
    }

    fast_mq_attr.nr_produce = sizeof(struct params_display);
    fast_mq_attr.nr_consume = sizeof(struct params_display);
    fast_mq_attr.nr_threshold = sizeof(struct params_display);
    CheckError(pgm_init_edge(&e5_6, collect_locations_node, display_node, "e5_6", &fast_mq_attr));

    pthread_barrier_t init_barrier;
    pthread_barrier_init(&init_barrier, 0, 3 * NUM_SCALE_LEVELS + 4);

    thread** t2 = (thread**) calloc(NUM_SCALE_LEVELS, sizeof(std::thread *));
    thread** t3 = (thread**) calloc(NUM_SCALE_LEVELS, sizeof(std::thread *));
    thread** t4 = (thread**) calloc(NUM_SCALE_LEVELS, sizeof(std::thread *));

    float bound_color_convert                   = 0;
    float bound_compute_scales                  = 0;
    float bounds_vxHOGCells  [NUM_SCALE_LEVELS] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    float bounds_normalize   [NUM_SCALE_LEVELS] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    float bounds_classify    [NUM_SCALE_LEVELS] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    //int bound_collect_locations               = 10;

    float cost_color_convert                   = 3.72;
    float cost_compute_scales                  = 2.09;
    float costs_vxHOGCells  [NUM_SCALE_LEVELS] = {0.11, 0.64, 0.75, 0.76, 0.86, 0.71, 0.61, 0.70, 0.70, 0.66, 0.60, 0.71, 1};
    float costs_normalize   [NUM_SCALE_LEVELS] = {13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
    float costs_classify    [NUM_SCALE_LEVELS] = {13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
    float cost_collect_locations               = 10;

    struct task_info t_info;
    //int period = PERIOD * args.num_fine_graphs;
    int m_cpus = 16;
    t_info.early = args.early;
    t_info.realtime = args.realtime;
    t_info.sched = coarse_unrolled;
    t_info.period = PERIOD;
    t_info.relative_deadline = PERIOD; //FAIR_LATENESS_PP(m_cpus, t_info.period, cost_compute_scales);
    t_info.phase = bound_color_convert;
    t_info.cluster = args.cluster;
    t_info.id = 1;
    thread t1(&cv::cuda::HOG::thread_fine_compute_scales, gpu_hog, &compute_scales_node, &init_barrier, t_info);
    for (int i=0; i<NUM_SCALE_LEVELS; i++) {
        t_info.relative_deadline = PERIOD; //FAIR_LATENESS_PP(m_cpus, t_info.period, costs_vxHOGCells[i]);
        t_info.id = 0 * NUM_SCALE_LEVELS + i + 2;
        t_info.phase = bound_color_convert + bound_compute_scales;
        t2[i] = new thread(&cv::cuda::HOG::thread_unrolled_vxHOGCells, gpu_hog, &unrolled_vxHOGCells_node[i], &init_barrier, t_info);

        t_info.relative_deadline = PERIOD; //FAIR_LATENESS_PP(m_cpus, t_info.period, costs_normalize[i]);
        t_info.id = 1 * NUM_SCALE_LEVELS + i + 2;
        t_info.phase = t_info.phase + bounds_vxHOGCells[i];
        t3[i] = new thread(&cv::cuda::HOG::thread_fine_normalize_histograms, gpu_hog, &normalize_node[i], &init_barrier, t_info);

        t_info.relative_deadline = PERIOD; //FAIR_LATENESS_PP(m_cpus, t_info.period, costs_classify[i]);
        t_info.id = 2 * NUM_SCALE_LEVELS + i + 2;
        t_info.phase = t_info.phase + bounds_normalize[i];
        t4[i] = new thread(&cv::cuda::HOG::thread_fine_classify, gpu_hog, &classify_node[i], &init_barrier, t_info);
    }
    t_info.relative_deadline = PERIOD; //FAIR_LATENESS_PP(m_cpus, t_info.period, cost_collect_locations);
    t_info.phase = t_info.phase + *std::max_element(bounds_classify, bounds_classify + NUM_SCALE_LEVELS);
    t_info.id = 3 * NUM_SCALE_LEVELS +  2;
    thread t5(&cv::cuda::HOG::thread_fine_collect_locations, gpu_hog, &collect_locations_node, &init_barrier, t_info);
    thread t6(&App::thread_display, this, &display_node, &init_barrier, args.display);

    pgm_claim_node(color_convert_node);

    edge_t *out_edge = (edge_t *)calloc(1, sizeof(edge_t));
    CheckError(pgm_get_edges_out(color_convert_node, out_edge, 1));
    struct params_compute *out_buf = (struct params_compute *)pgm_get_edge_buf_p(*out_edge);
    if (out_buf == NULL)
        fprintf(stderr, "color convert out buffer is NULL\n");

    Size win_stride(args.win_stride_width, args.win_stride_height);
    Size win_size(args.win_width, args.win_width * 2);

    Mat img_aux;
    Mat* img = new Mat();
    Mat* img_to_show;
    cuda::GpuMat* gpu_img = new cuda::GpuMat();
    vector<Rect>* found = new vector<Rect>();
    Mat frame;
    cv::cuda::Stream stream;

    pthread_barrier_wait(&init_barrier);

    t_info.id = 0;
    if (args.realtime) {
        struct rt_task param;
        t_info.relative_deadline = PERIOD; //FAIR_LATENESS_PP(m_cpus, t_info.period, cost_color_convert);
        t_info.phase = 0;

        if (t_info.cluster != -1)
            CALL(be_migrate_to_domain(t_info.cluster));
        init_rt_task_param(&param);
        param.exec_cost = ms2ns(EXEC_COST);
        param.period = ms2ns(t_info.period);
        param.relative_deadline = ms2ns(t_info.relative_deadline);
        param.budget_policy = NO_ENFORCEMENT;
        param.cls = RT_CLASS_SOFT;
        param.priority = LITMUS_LOWEST_PRIORITY;
        param.cpu = domain_to_first_cpu(t_info.cluster);
        if (t_info.cluster != -1)
            param.cpu = domain_to_first_cpu(t_info.cluster);
        CALL( init_litmus() );
        CALL( set_rt_task_param(gettid(), &param) );
        CALL( task_mode(LITMUS_RT_TASK) );
        CALL( wait_for_ts_release() );
    }

    int count_frame = 0;
    while (count_frame < args.count && running) {

        for (int j=0; j<100; j++) {
            if (!args.realtime)
                usleep(30000);
            if (count_frame >= args.count)
                break;
            frame = frames[j];
            workBegin();

            // Change format of the image
            if (make_gray) cvtColor(frame, img_aux, COLOR_BGR2GRAY);
            else if (use_gpu) cvtColor(frame, img_aux, COLOR_BGR2BGRA);
            else frame.copyTo(img_aux);

            // Resize image
            if (args.resize_src) resize(img_aux, *img, Size(args.width, args.height));
            else *img = img_aux;
            img_to_show = img;

            // Perform HOG classification
            hogWorkBegin();
            if (use_gpu) {
                gpu_img->upload(*img, stream);
                cudaStreamSynchronize(cv::cuda::StreamAccessor::getStream(stream));
                gpu_hog->setNumLevels(nlevels);
                gpu_hog->setHitThreshold(hit_threshold);
                gpu_hog->setScaleFactor(scale);
                gpu_hog->setGroupThreshold(gr_threshold);

                out_buf->gpu_img = gpu_img;
                out_buf->found = found;
                out_buf->img_to_show = img;
                out_buf->frame_index = count_frame;
                out_buf->start_time = hog_work_begin;

                CheckError(pgm_complete(color_convert_node));
            } else {
                cpu_hog.nlevels = nlevels;
                cpu_hog.detectMultiScale(*img, *found, hit_threshold, win_stride,
                        Size(0, 0), scale, gr_threshold);
                hogWorkEnd();

                // Draw positive classified windows
                for (size_t i = 0; i < found->size(); i++) {
                    Rect r = (*found)[i];
                    rectangle(*img_to_show, r.tl(), r.br(), Scalar(0, 255, 0), 3);
                }

                if (use_gpu)
                    putText(*img_to_show, "Mode: GPU", Point(5, 25), FONT_HERSHEY_SIMPLEX, 1., Scalar(255, 100, 0), 2);
                else
                    putText(*img_to_show, "Mode: CPU", Point(5, 25), FONT_HERSHEY_SIMPLEX, 1., Scalar(255, 100, 0), 2);
                putText(*img_to_show, "FPS HOG: " + hogWorkFps(), Point(5, 65), FONT_HERSHEY_SIMPLEX, 1., Scalar(255, 100, 0), 2);
                putText(*img_to_show, "FPS total: " + workFps(), Point(5, 105), FONT_HERSHEY_SIMPLEX, 1., Scalar(255, 100, 0), 2);
                imshow("opencv_gpu_hog", *img_to_show);

                workEnd();

                if (args.write_video) {
                    if (!video_writer.isOpened()) {
                        video_writer.open(args.dst_video, VideoWriter::fourcc('x','v','i','d'), args.dst_video_fps,
                                img_to_show->size(), true);
                        if (!video_writer.isOpened())
                            throw std::runtime_error("can't create video writer");
                    }

                    if (make_gray) cvtColor(*img_to_show, *img, COLOR_GRAY2BGR);
                    else cvtColor(*img_to_show, *img, COLOR_BGRA2BGR);

                    video_writer << *img;
                }

                handleKey((char)waitKey(3));
                delete gpu_img;
                delete found;
                delete img;
            }

            gpu_img = new cuda::GpuMat();
            found = new vector<Rect>();
            img = new Mat();
            count_frame++;
            /* Wait until the next job is released. */
            if (args.realtime)
                sleep_next_period();
        }
    }
    /*****
     * 6) Transition to background mode.
     */

    free(out_edge);
    CheckError(pgm_terminate(color_convert_node));
    pthread_barrier_wait(&init_barrier);
    CheckError(pgm_release_node(color_convert_node));
    printf("Joining pthreads...\n");
    t1.join();
    for (int i=0; i<NUM_SCALE_LEVELS; i++) {
        if (t2[i]->joinable()) t2[i]->join();
        if (t3[i]->joinable()) t3[i]->join();
        if (t4[i]->joinable()) t4[i]->join();
        delete t2[i];
        delete t3[i];
        delete t4[i];
    }
    delete t2;
    delete t3;
    delete t4;
    t5.join();
    t6.join();

    CheckError(pgm_destroy_graph(g));
    CheckError(pgm_destroy());
    if (args.realtime)
        CALL( task_mode(BACKGROUND_TASK) );
}

void App::thread_color_convert(node_t *_node, pthread_barrier_t* init_barrier,
        cv::Ptr<cv::cuda::HOG> gpu_hog, cv::HOGDescriptor cpu_hog, Mat* frames, struct task_info t_info, int graph_idx)
{
    fprintf(stdout, "node name: color_convert(source), task id: %d, node tid: %d\n", t_info.id, gettid());
    node_t node = *_node;
#ifdef LOG_DEBUG
    char tabbuf[] = "\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t";
    tabbuf[node.node] = '\0';
#endif
    pgm_claim_node(node);

    edge_t *out_edge = (edge_t *)calloc(1, sizeof(edge_t));
    CheckError(pgm_get_edges_out(node, out_edge, 1));
    struct params_compute *out_buf = (struct params_compute *)pgm_get_edge_buf_p(*out_edge);
    if (out_buf == NULL)
        fprintf(stderr, "color convert out buffer is NULL\n");

    Size win_stride(args.win_stride_width, args.win_stride_height);
    Size win_size(args.win_width, args.win_width * 2);

    Mat img_aux;
    Mat* img = new Mat();
    Mat* img_to_show;
    cuda::GpuMat* gpu_img = new cuda::GpuMat();
    vector<Rect>* found = new vector<Rect>();
    Mat frame;

    /* initialization is finished */
    pthread_barrier_wait(init_barrier);

    if (t_info.realtime) {
        if (t_info.cluster != -1)
            CALL(be_migrate_to_domain(t_info.cluster));
        struct rt_task param;
        init_rt_task_param(&param);
        param.exec_cost = ms2ns(EXEC_COST);
        param.period = ms2ns(t_info.period);
        param.relative_deadline = ms2ns(t_info.relative_deadline);
        param.phase = ms2ns(t_info.phase);
        param.budget_policy = NO_ENFORCEMENT;
        param.cls = RT_CLASS_SOFT;
        param.priority = LITMUS_LOWEST_PRIORITY;
        if (t_info.cluster != -1)
            param.cpu = domain_to_first_cpu(t_info.cluster);
        CALL( init_litmus() );
        CALL( set_rt_task_param(gettid(), &param) );
        CALL( task_mode(LITMUS_RT_TASK) );
        CALL( wait_for_ts_release() );
    }

    int count_frame = 0;
    while (count_frame < args.count / args.num_fine_graphs && running) {
        for (int j = graph_idx; j < 100; j += args.num_fine_graphs) {
            if (!t_info.realtime)
                usleep(30000);
            if (count_frame >= args.count)
                break;
            frame = frames[j];
            workBegin();

            /* color convert node starts below */
            // Change format of the image
            if (make_gray) cvtColor(frame, img_aux, COLOR_BGR2GRAY);
            else if (use_gpu) cvtColor(frame, img_aux, COLOR_BGR2BGRA);
            else frame.copyTo(img_aux);

            // Resize image
            if (args.resize_src) resize(img_aux, *img, Size(args.width, args.height));
            else *img = img_aux;
            img_to_show = img;

            // Perform HOG classification
            hogWorkBegin();
            cv::cuda::Stream stream;
            if (use_gpu) {
                gpu_img->upload(*img, stream);
                cudaStreamSynchronize(cv::cuda::StreamAccessor::getStream(stream));
                gpu_hog->setNumLevels(nlevels);
                gpu_hog->setHitThreshold(hit_threshold);
                gpu_hog->setScaleFactor(scale);
                gpu_hog->setGroupThreshold(gr_threshold);
                out_buf->gpu_img = gpu_img;
                out_buf->found = found;
                out_buf->img_to_show = img;
                out_buf->frame_index = j;
                out_buf->start_time = hog_work_begin;
                CheckError(pgm_complete(node));
            } else {
                cpu_hog.nlevels = nlevels;
                cpu_hog.detectMultiScale(*img, *found, hit_threshold, win_stride,
                        Size(0, 0), scale, gr_threshold);
                hogWorkEnd();

                // Draw positive classified windows
                for (size_t i = 0; i < found->size(); i++) {
                    Rect r = (*found)[i];
                    rectangle(*img_to_show, r.tl(), r.br(), Scalar(0, 255, 0), 3);
                }

                if (use_gpu)
                    putText(*img_to_show, "Mode: GPU", Point(5, 25), FONT_HERSHEY_SIMPLEX, 1., Scalar(255, 100, 0), 2);
                else
                    putText(*img_to_show, "Mode: CPU", Point(5, 25), FONT_HERSHEY_SIMPLEX, 1., Scalar(255, 100, 0), 2);
                putText(*img_to_show, "FPS HOG: " + hogWorkFps(), Point(5, 65), FONT_HERSHEY_SIMPLEX, 1., Scalar(255, 100, 0), 2);
                putText(*img_to_show, "FPS total: " + workFps(), Point(5, 105), FONT_HERSHEY_SIMPLEX, 1., Scalar(255, 100, 0), 2);
                imshow("opencv_gpu_hog", *img_to_show);

                workEnd();

                if (args.write_video) {
                    if (!video_writer.isOpened()) {
                        video_writer.open(args.dst_video,
                                VideoWriter::fourcc('x','v','i','d'),
                                args.dst_video_fps, img_to_show->size(), true);
                        if (!video_writer.isOpened())
                            throw std::runtime_error("can't create video writer");
                    }

                    if (make_gray) cvtColor(*img_to_show, *img, COLOR_GRAY2BGR);
                    else cvtColor(*img_to_show, *img, COLOR_BGRA2BGR);

                    video_writer << *img;
                }

                handleKey((char)waitKey(3));
                delete gpu_img;
                delete found;
                delete img;
            }

            gpu_img = new cuda::GpuMat();
            found = new vector<Rect>();
            img = new Mat();
            count_frame++;
            if (t_info.realtime)
                sleep_next_period();
        }
    }

    free(out_edge);
    CheckError(pgm_terminate(node));

    pthread_barrier_wait(init_barrier);

    CheckError(pgm_release_node(node));

    if (args.realtime)
        CALL( task_mode(BACKGROUND_TASK) );
}

static void sync_info_init(struct sync_info *s)
{
    s->start_time = 0;
    s->job_no = 0;
}


void App::sched_fine_grained_hog(cv::Ptr<cv::cuda::HOG> gpu_hog, cv::HOGDescriptor cpu_hog, Mat* frames)
{
    pthread_barrier_t arr_fine_init_barrier[args.num_fine_graphs];
    /* graph construction */
    graph_t arr_g [args.num_fine_graphs];

    struct sync_info arr_sync_info_resize             [args.num_fine_graphs][NUM_SCALE_LEVELS];
    struct sync_info arr_sync_info_compute_gradients  [args.num_fine_graphs][NUM_SCALE_LEVELS];
    struct sync_info arr_sync_info_compute_histograms [args.num_fine_graphs][NUM_SCALE_LEVELS];
    struct sync_info arr_sync_info_normalize          [args.num_fine_graphs][NUM_SCALE_LEVELS];
    struct sync_info arr_sync_info_classify           [args.num_fine_graphs][NUM_SCALE_LEVELS];

    thread** arr_t0 = (thread**) calloc(args.num_fine_graphs, sizeof(std::thread *));
    thread** arr_t1 = (thread**) calloc(args.num_fine_graphs, sizeof(std::thread *));
    thread** arr_t2 = (thread**) calloc(args.num_fine_graphs * NUM_SCALE_LEVELS, sizeof(std::thread *));
    thread** arr_t3 = (thread**) calloc(args.num_fine_graphs * NUM_SCALE_LEVELS, sizeof(std::thread *));
    thread** arr_t4 = (thread**) calloc(args.num_fine_graphs * NUM_SCALE_LEVELS, sizeof(std::thread *));
    thread** arr_t5 = (thread**) calloc(args.num_fine_graphs * NUM_SCALE_LEVELS, sizeof(std::thread *));
    thread** arr_t6 = (thread**) calloc(args.num_fine_graphs * NUM_SCALE_LEVELS, sizeof(std::thread *));
    thread** arr_t7 = (thread**) calloc(args.num_fine_graphs, sizeof(std::thread *));
    thread** arr_t8 = (thread**) calloc(args.num_fine_graphs, sizeof(std::thread *));

    char buf[30];
    sprintf(buf, "/tmp/graph_t%d", args.task_id);
    CheckError(pgm_init(buf, 1));

    for (int g_idx = 0; g_idx < args.num_fine_graphs; g_idx++)
    {
        pthread_barrier_t* fine_init_barrier = arr_fine_init_barrier + g_idx;

        // A graph consists of the graph itself, nodes, and edge
        graph_t* g_ptr = arr_g + g_idx;

        node_t color_convert_node;
        node_t compute_scales_node;
        node_t resize_node             [NUM_SCALE_LEVELS];
        node_t compute_gradients_node  [NUM_SCALE_LEVELS];
        node_t compute_histograms_node [NUM_SCALE_LEVELS];
        node_t normalize_node          [NUM_SCALE_LEVELS];
        node_t classify_node           [NUM_SCALE_LEVELS];
        node_t collect_locations_node;
        node_t display_node;

        edge_t e0_1;
        edge_t e1_2 [NUM_SCALE_LEVELS];
        edge_t e2_3 [NUM_SCALE_LEVELS];
        edge_t e3_4 [NUM_SCALE_LEVELS];
        edge_t e4_5 [NUM_SCALE_LEVELS];
        edge_t e5_6 [NUM_SCALE_LEVELS];
        edge_t e6_7 [NUM_SCALE_LEVELS];
        edge_t e7_8;

        // Initialize the graph
        sprintf(buf, "hog_%d", g_idx);
        CheckError(pgm_init_graph(g_ptr, buf));
        graph_t g = *g_ptr;

        // Initialize the nodes
        CheckError(pgm_init_node(&color_convert_node, g, "color_convert"));
        CheckError(pgm_init_node(&compute_scales_node, g, "compute_scales"));
        for (int i=0; i<NUM_SCALE_LEVELS; i++) {
            CheckError(pgm_init_node(resize_node + i, g, "resize"));
            CheckError(pgm_init_node(compute_gradients_node + i, g, "compute_gradients"));
            CheckError(pgm_init_node(compute_histograms_node + i, g, "compute_histograms"));
            CheckError(pgm_init_node(normalize_node + i, g, "normalize"));
            CheckError(pgm_init_node(classify_node + i, g, "classify"));
        }
        CheckError(pgm_init_node(&collect_locations_node, g, "collect_locations"));
        CheckError(pgm_init_node(&display_node, g, "display"));

        // Initialize the edges
        edge_attr_t fast_mq_attr;
        memset(&fast_mq_attr, 0, sizeof(fast_mq_attr));
        //fast_mq_attr.mq_maxmsg = 20; /* root required for values larger than 10 */
        //fast_mq_attr.type = pgm_fast_mq_edge;
        fast_mq_attr.type = pgm_fast_fifo_edge;

        fast_mq_attr.nr_produce = sizeof(struct params_compute);
        fast_mq_attr.nr_consume = sizeof(struct params_compute);
        fast_mq_attr.nr_threshold = sizeof(struct params_compute);
        CheckError(pgm_init_edge(&e0_1, color_convert_node, compute_scales_node, "e0_1", &fast_mq_attr));

        for (int i=0; i<NUM_SCALE_LEVELS; i++) {
            sprintf(buf, "e1_2_%d", i);
            fast_mq_attr.nr_produce = sizeof(struct params_resize);
            fast_mq_attr.nr_consume = sizeof(struct params_resize);
            fast_mq_attr.nr_threshold = sizeof(struct params_resize);
            CheckError(pgm_init_edge(e1_2 + i,
                        compute_scales_node,
                        resize_node[i], buf, &fast_mq_attr));

            sprintf(buf, "e2_3_%d", i);
            fast_mq_attr.nr_produce = sizeof(struct params_compute_gradients);
            fast_mq_attr.nr_consume = sizeof(struct params_compute_gradients);
            fast_mq_attr.nr_threshold = sizeof(struct params_compute_gradients);
            CheckError(pgm_init_edge(e2_3 + i,
                        resize_node[i],
                        compute_gradients_node[i], buf, &fast_mq_attr));

            sprintf(buf, "e3_4_%d", i);
            fast_mq_attr.nr_produce = sizeof(struct params_compute_histograms);
            fast_mq_attr.nr_consume = sizeof(struct params_compute_histograms);
            fast_mq_attr.nr_threshold = sizeof(struct params_compute_histograms);
            CheckError(pgm_init_edge(e3_4 + i,
                        compute_gradients_node[i],
                        compute_histograms_node[i], buf, &fast_mq_attr));

            sprintf(buf, "e4_5_%d", i);
            fast_mq_attr.nr_produce = sizeof(struct params_fine_normalize);
            fast_mq_attr.nr_consume = sizeof(struct params_fine_normalize);
            fast_mq_attr.nr_threshold = sizeof(struct params_fine_normalize);
            CheckError(pgm_init_edge(e4_5 + i,
                        compute_histograms_node[i],
                        normalize_node[i], buf, &fast_mq_attr));

            sprintf(buf, "e5_6_%d", i);
            fast_mq_attr.nr_produce = sizeof(struct params_fine_classify);
            fast_mq_attr.nr_consume = sizeof(struct params_fine_classify);
            fast_mq_attr.nr_threshold = sizeof(struct params_fine_classify);
            CheckError(pgm_init_edge(e5_6 + i,
                        normalize_node[i],
                        classify_node[i], buf, &fast_mq_attr));

            sprintf(buf, "e6_7_%d", i);
            fast_mq_attr.nr_produce = sizeof(struct params_fine_collect_locations);
            fast_mq_attr.nr_consume = sizeof(struct params_fine_collect_locations);
            fast_mq_attr.nr_threshold = sizeof(struct params_fine_collect_locations);
            CheckError(pgm_init_edge(e6_7 + i,
                        classify_node[i],
                        collect_locations_node, buf, &fast_mq_attr));
        }

        fast_mq_attr.nr_produce = sizeof(struct params_display);
        fast_mq_attr.nr_consume = sizeof(struct params_display);
        fast_mq_attr.nr_threshold = sizeof(struct params_display);
        CheckError(pgm_init_edge(&e7_8, collect_locations_node, display_node, "e7_8", &fast_mq_attr));

        pthread_barrier_init(fine_init_barrier, 0, 5 * NUM_SCALE_LEVELS + 4);

        thread** t0 = arr_t0 + g_idx;
        thread** t1 = arr_t1 + g_idx;
        thread** t2 = arr_t2 + g_idx * NUM_SCALE_LEVELS;
        thread** t3 = arr_t3 + g_idx * NUM_SCALE_LEVELS;
        thread** t4 = arr_t4 + g_idx * NUM_SCALE_LEVELS;
        thread** t5 = arr_t5 + g_idx * NUM_SCALE_LEVELS;
        thread** t6 = arr_t6 + g_idx * NUM_SCALE_LEVELS;
        thread** t7 = arr_t7 + g_idx;
        thread** t8 = arr_t8 + g_idx;

        struct sync_info* in_sync_info_resize             = arr_sync_info_resize            [((g_idx + args.num_fine_graphs - 1) % args.num_fine_graphs)];
        struct sync_info* in_sync_info_compute_gradients  = arr_sync_info_compute_gradients [((g_idx + args.num_fine_graphs - 1) % args.num_fine_graphs)];
        struct sync_info* in_sync_info_compute_histograms = arr_sync_info_compute_histograms[((g_idx + args.num_fine_graphs - 1) % args.num_fine_graphs)];
        struct sync_info* in_sync_info_normalize          = arr_sync_info_normalize         [((g_idx + args.num_fine_graphs - 1) % args.num_fine_graphs)];
        struct sync_info* in_sync_info_classify           = arr_sync_info_classify          [((g_idx + args.num_fine_graphs - 1) % args.num_fine_graphs)];

        struct sync_info* out_sync_info_resize             = arr_sync_info_resize            [g_idx];
        struct sync_info* out_sync_info_compute_gradients  = arr_sync_info_compute_gradients [g_idx];
        struct sync_info* out_sync_info_compute_histograms = arr_sync_info_compute_histograms[g_idx];
        struct sync_info* out_sync_info_normalize          = arr_sync_info_normalize         [g_idx];
        struct sync_info* out_sync_info_classify           = arr_sync_info_classify          [g_idx];

        for (int i=0; i<NUM_SCALE_LEVELS; i++) {
            sync_info_init(i + in_sync_info_resize);
            sync_info_init(i + in_sync_info_compute_gradients);
            sync_info_init(i + in_sync_info_compute_histograms);
            sync_info_init(i + in_sync_info_normalize);
            sync_info_init(i + in_sync_info_classify);

            sync_info_init(i + out_sync_info_resize);
            sync_info_init(i + out_sync_info_compute_gradients);
            sync_info_init(i + out_sync_info_compute_histograms);
            sync_info_init(i + out_sync_info_normalize);
            sync_info_init(i + out_sync_info_classify);
        }

        /* G-FL fine-grained HOG */
        float bound_color_convert                   = 31.0194960215;
        float bound_compute_scales                  = 30.5688293615;
        float bounds_resize      [NUM_SCALE_LEVELS] = {30.2419972231,30.3148010256,30.2936775569,30.3079650324,30.2963991483,30.3058228698,30.3005336032,30.2888197835,30.293491626,30.2840584057,30.3013304497,30.3016323015,30.3111116087};
        float bounds_compute_grad[NUM_SCALE_LEVELS] = {30.4902863291,30.44552396,30.4706995604,30.4291495557,30.4171804989,30.4135009326,30.3785928987,30.4059076714,30.3816881109,30.3627509567,30.3564101342,30.3519205285,30.3405616838};
        float bounds_compute_hist[NUM_SCALE_LEVELS] = {30.5326861275,30.4951636278,30.5415047381,30.4958614402,30.5149987735,30.4750125581,30.5373623676,30.522424222,30.4851937235,30.4577396346,30.4263855713,30.4118414514,30.5202006157};
        float bounds_normalize   [NUM_SCALE_LEVELS] = {30.2903383654,30.3080869341,30.2928467608,30.3064130287,30.3114078316,30.3120987837,30.2985990076,30.2803770804,30.3099685825,30.3123524377,30.2989110618,30.2943410679,30.3081075149};
        float bounds_classify    [NUM_SCALE_LEVELS] = {30.4792191337,30.4948973087,30.4540269983,30.429525815,30.5229441951,30.5070612675,30.497249677,30.4955912512,30.4267037822,30.4508744413,30.430038752,30.3991297266,30.3767439685};
        //float bound_collect_locations               30.983976896;

        /* G-FL fine-grained HOG */
        float cost_color_convert                   = 4.509346;
        float cost_compute_scales                  = 1.947347;
        float costs_resize      [NUM_SCALE_LEVELS] = {0.089336,0.503219,0.383134,0.464357,0.398606,0.452179,0.42211,0.355518,0.382077,0.32845,0.42664,0.428356,0.482245};
        float costs_compute_grad[NUM_SCALE_LEVELS] = {1.500837,1.246367,1.389488,1.15328,1.085237,1.064319,0.86587,1.021152,0.883466,0.77581,0.739763,0.71424,0.649666};
        float costs_compute_hist[NUM_SCALE_LEVELS] = {1.741876,1.528564,1.792009,1.532531,1.641325,1.414007,1.76846,1.683538,1.471886,1.315812,1.137567,1.054885,1.670897};
        float costs_normalize   [NUM_SCALE_LEVELS] = {0.364151,0.46505,0.378411,0.455534,0.483929,0.487857,0.411112,0.307522,0.475747,0.489299,0.412886,0.386906,0.465167};
        float costs_classify    [NUM_SCALE_LEVELS] = {1.437921,1.52705,1.294706,1.155419,1.686494,1.596201,1.540423,1.530995,1.139376,1.276784,1.158335,0.98262,0.855359};
        float cost_collect_locations               = 4.307423;

        /* C-FL fine-grained HOG */
        //float bound_color_convert                   = 26.3496352098;
        //float bound_compute_scales                  = 25.6841982494;
        //float bounds_resize      [NUM_SCALE_LEVELS] = {25.0939476981,25.1387935171,25.130189875,25.1357033275,25.1312621102,25.1463332249,25.1316714257,25.1302245094,25.1267072576,25.1262401226,25.1348909935,25.1432006726,25.1413123823};
        //float bounds_compute_grad[NUM_SCALE_LEVELS] = {25.3489965402,25.4185896374,25.4092134479,25.4079932294,25.4051932818,25.4030496702,25.312479578,25.291207475,25.2868561354,25.2685316772,25.2518075571,25.246440943,25.2364255915};
        //float bounds_compute_hist[NUM_SCALE_LEVELS] = {25.5728014354,25.4973631396,25.5123077387,25.471628638,25.3841716342,25.3274278981,25.4720253593,25.5328196023,25.4654399579,25.3426074958,25.4860697502,25.3209827517,25.4610373822};
        //float bounds_normalize   [NUM_SCALE_LEVELS] = {25.1658996554,25.1786894798,25.1487118351,25.1680595823,25.1750299693,25.1885187788,25.1646279147,25.1649092834,25.1582577618,25.1765865135,25.1940356661,25.1748711091,25.179600851};
        //float bounds_classify    [NUM_SCALE_LEVELS] = {25.2909527262,25.3017792668,25.4537335318,25.3216511097,25.4693284561,25.3272781973,25.3334016159,25.3265829333,25.3061011243,25.2757711248,25.2811148401,25.2748319888,25.3276382806};
        ////float bound_collect_locations               26.204139258;

        /////* C-FL fine-grained HOG */
        //float cost_color_convert                   = 4.475663;
        //float cost_compute_scales                  = 2.150868;
        //float costs_resize      [NUM_SCALE_LEVELS] = {0.088747,0.245422,0.215364,0.234626,0.21911,0.271763,0.22054,0.215485,0.203197,0.201565,0.231788,0.260819,0.254222};
        //float costs_compute_grad[NUM_SCALE_LEVELS] = {0.979795,1.222928,1.190171,1.185908,1.176126,1.168637,0.852218,0.777901,0.762699,0.69868,0.640252,0.621503,0.586513};
        //float costs_compute_hist[NUM_SCALE_LEVELS] = {1.761688,1.498134,1.550345,1.408227,1.102684,0.904442,1.409613,1.622006,1.386606,0.957474,1.458679,0.881925,1.371225};
        //float costs_normalize   [NUM_SCALE_LEVELS] = {0.340121,0.384804,0.280073,0.347667,0.372019,0.419144,0.335678,0.336661,0.313423,0.377457,0.438418,0.371464,0.387988};
        //float costs_classify    [NUM_SCALE_LEVELS] = {0.777011,0.814835,1.345708,0.88426,1.400191,0.903919,0.925312,0.90149,0.829934,0.723972,0.742641,0.720691,0.905177};
        //float cost_collect_locations               = 3.967353;

        /* | first graph release      | second graph release     | first graph release again
         *  <---------PERIOD--------->
         *  <--------------- PERIOD * args.num_fine_graphs ---------->
         */
        int period = PERIOD * args.num_fine_graphs;
        int m_cpus = 16;
        struct task_info t_info;
        t_info.early = args.early;
        t_info.realtime = args.realtime;
        t_info.sched = fine_grained;
        t_info.period = period;
        t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, cost_color_convert);
        t_info.phase = PERIOD * g_idx;
        // t_info.fzlp_nodes = (fzlp_node_t *)calloc(NUM_SCALE_LEVELS, sizeof(fzlp_node_t));
        t_info.id = 0;
        if (args.cluster != -1)
            t_info.cluster = args.cluster;// + g_idx;
        else
            t_info.cluster = args.cluster;
        *t0 = new thread(&App::thread_color_convert, this,
                &color_convert_node, fine_init_barrier,
                gpu_hog, cpu_hog, frames, t_info, g_idx);

        t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, cost_compute_scales);
        t_info.id = 1;
        t_info.phase = t_info.phase + bound_color_convert;
        *t1 = new thread(&cv::cuda::HOG::thread_fine_compute_scales, gpu_hog,
                &compute_scales_node, fine_init_barrier, t_info);

        for (int i=0; i<NUM_SCALE_LEVELS; i++) {
            t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, costs_resize[i]);
            t_info.id = 0 * NUM_SCALE_LEVELS + i + 2;
            t_info.phase = PERIOD * g_idx + bound_color_convert + bound_compute_scales;
            t_info.s_info_in = in_sync_info_resize + i;
            t_info.s_info_out = out_sync_info_resize + i;
            t2[i] = new thread(&cv::cuda::HOG::thread_fine_resize, gpu_hog,
                    resize_node + i, fine_init_barrier, t_info);

            t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, costs_compute_grad[i]);
            t_info.id = 1 * NUM_SCALE_LEVELS + i + 2;
            t_info.phase = t_info.phase + bounds_resize[i];
            t_info.s_info_in = in_sync_info_compute_gradients + i;
            t_info.s_info_out = out_sync_info_compute_gradients + i;
            t3[i] = new thread(&cv::cuda::HOG::thread_fine_compute_gradients,
                    gpu_hog, compute_gradients_node + i, fine_init_barrier, t_info);

            t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, costs_compute_hist[i]);
            t_info.id = 2 * NUM_SCALE_LEVELS + i + 2;
            t_info.phase = t_info.phase + bounds_compute_grad[i];
            t_info.s_info_in = in_sync_info_compute_histograms + i;
            t_info.s_info_out = out_sync_info_compute_histograms + i;
            t4[i] = new thread(&cv::cuda::HOG::thread_fine_compute_histograms,
                    gpu_hog, compute_histograms_node + i, fine_init_barrier,
                    t_info);

            t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, costs_normalize[i]);
            t_info.id = 3 * NUM_SCALE_LEVELS + i + 2;
            t_info.phase = t_info.phase + bounds_compute_hist[i];
            t_info.s_info_in = in_sync_info_normalize + i;
            t_info.s_info_out = out_sync_info_normalize + i;
            t5[i] = new thread(&cv::cuda::HOG::thread_fine_normalize_histograms,
                    gpu_hog, normalize_node + i, fine_init_barrier, t_info);

            t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, costs_classify[i]);
            t_info.id = 4 * NUM_SCALE_LEVELS + i + 2;
            t_info.phase = t_info.phase + bounds_normalize[i];
            t_info.s_info_in = in_sync_info_classify + i;
            t_info.s_info_out = out_sync_info_classify + i;
            t6[i] = new thread(&cv::cuda::HOG::thread_fine_classify, gpu_hog,
                    classify_node + i, fine_init_barrier, t_info);
        }

        t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, cost_collect_locations);
        t_info.id = 5 * NUM_SCALE_LEVELS + 2;
        t_info.phase = t_info.phase + *std::max_element(bounds_classify, bounds_classify+NUM_SCALE_LEVELS);
        *t7 = new thread(&cv::cuda::HOG::thread_fine_collect_locations, gpu_hog,
                &collect_locations_node, fine_init_barrier, t_info);
        //t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, cost_display);
        //t_info.phase;
        *t8 = new thread(&App::thread_display, this, &display_node,
                fine_init_barrier, g_idx == 0 && args.display);
    }

    /* graph construction finishes */

    printf("Joining pthreads...\n");

    for (int g_idx = 0; g_idx < args.num_fine_graphs; g_idx++) {
        graph_t g = arr_g[g_idx];
        thread* t0 = arr_t0[g_idx];
        thread* t1 = arr_t1[g_idx];
        thread** t2 = arr_t2 + g_idx * NUM_SCALE_LEVELS;
        thread** t3 = arr_t3 + g_idx * NUM_SCALE_LEVELS;
        thread** t4 = arr_t4 + g_idx * NUM_SCALE_LEVELS;
        thread** t5 = arr_t5 + g_idx * NUM_SCALE_LEVELS;
        thread** t6 = arr_t6 + g_idx * NUM_SCALE_LEVELS;
        thread* t7 = arr_t7[g_idx];
        thread* t8 = arr_t8[g_idx];
        t0->join();
        t1->join();
        delete t0;
        delete t1;
        for (int i=0; i<NUM_SCALE_LEVELS; i++) {
            if (t2[i]->joinable()) t2[i]->join();
            if (t3[i]->joinable()) t3[i]->join();
            if (t4[i]->joinable()) t4[i]->join();
            if (t5[i]->joinable()) t5[i]->join();
            if (t6[i]->joinable()) t6[i]->join();

            delete t2[i];
            delete t3[i];
            delete t4[i];
            delete t5[i];
            delete t6[i];
        }
        t7->join();
        t8->join();
        delete t7;
        delete t8;
        CheckError(pgm_destroy_graph(g));
    }
    free(arr_t1);
    free(arr_t2);
    free(arr_t3);
    free(arr_t4);
    free(arr_t5);
    free(arr_t6);
    free(arr_t7);
    free(arr_t8);

    //CheckError(pgm_destroy_graph(g));
    CheckError(pgm_destroy());
    fprintf(stdout, "cleaned up ...");
}

void App::sched_etoe_hog_preload(cv::Ptr<cv::cuda::HOG> gpu_hog, cv::HOGDescriptor cpu_hog, Mat* frames)
{
    fprintf(stdout, "node name: color_convert(source), task id: 0, node tid: %d\n", gettid());
    fprintf(stdout, "node name: collect_locations(sink), task id: 1, node tid: %d\n", gettid());

    Size win_stride(args.win_stride_width, args.win_stride_height);
    Size win_size(args.win_width, args.win_width * 2);

    Mat img_aux;
    Mat* img = new Mat();
    Mat* img_to_show;
    cuda::GpuMat* gpu_img = new cuda::GpuMat();
    vector<Rect>* found = new vector<Rect>();

    Mat frame;
    int count_frame = 0;
    cudaFree(0);

    if (args.realtime) {
        if (args.cluster != -1)
            CALL(be_migrate_to_domain(args.cluster));
        struct rt_task param;

        /* Setup task parameters */
        init_rt_task_param(&param);
        param.exec_cost = ms2ns(EXEC_COST);
        param.period = ms2ns(PERIOD);
        //param.relative_deadline = ms2ns(RELATIVE_DEADLINE);
        param.relative_deadline = ms2ns(FAIR_LATENESS_PP(32, param.period, param.exec_cost));
        if (args.cluster != -1)
            param.cpu = domain_to_first_cpu(args.cluster);

        /* What to do in the case of budget overruns? */
        param.budget_policy = NO_ENFORCEMENT;

        /* The task class parameter is ignored by most plugins. */
        param.cls = RT_CLASS_SOFT;

        /* The priority parameter is only used by fixed-priority plugins. */
        param.priority = LITMUS_LOWEST_PRIORITY;

        /*****
         * 3) Setup real-time parameters.
         *    In this example, we create a sporadic task that does not specify a
         *    target partition (and thus is intended to run under global scheduling).
         *    If this were to execute under a partitioned scheduler, it would be assigned
         *    to the first partition (since partitioning is performed offline).
         */
        CALL( init_litmus() );

        /* To specify a partition, do
         *
         * param.cpu = CPU;
         * be_migrate_to(CPU);
         *
         * where CPU ranges from 0 to "Number of CPUs" - 1 before calling
         * set_rt_task_param().
         */
        CALL( set_rt_task_param(gettid(), &param) );
        fprintf(stdout, "threa id %d is now real-time task\n", gettid());

        /*****
         * 4) Transition to real-time mode.
         */
        CALL( task_mode(LITMUS_RT_TASK) );
        CALL( wait_for_ts_release() );
    }

    /* The task is now executing as a real-time task if the call didn't fail.
     */
    while (count_frame < args.count && running) {
        for (int j=0; j<100; j++) {
            if (count_frame >= args.count)
                break;
            frame = frames[j];
            if (!args.realtime)
                usleep(33000);
            workBegin();

            // Change format of the image
            if (make_gray) cvtColor(frame, img_aux, COLOR_BGR2GRAY);
            else if (use_gpu) cvtColor(frame, img_aux, COLOR_BGR2BGRA);
            else frame.copyTo(img_aux);

            // Resize image
            if (args.resize_src) resize(img_aux, *img, Size(args.width, args.height));
            else *img = img_aux;
            img_to_show = img;

            // Perform HOG classification
            hogWorkBegin();
            if (use_gpu) {
                gpu_img->upload(*img);
                gpu_hog->setNumLevels(nlevels);
                gpu_hog->setHitThreshold(hit_threshold);
                gpu_hog->setScaleFactor(scale);
                gpu_hog->setGroupThreshold(gr_threshold);
                gpu_hog->detectMultiScale(*gpu_img, *found);
            } else {
                cpu_hog.nlevels = nlevels;
                cpu_hog.detectMultiScale(*img, *found, hit_threshold, win_stride,
                        Size(0, 0), scale, gr_threshold);
            }
            hogWorkEnd();

            printf("%d response time: %f\n", count_frame, 1/hog_work_fps);
            // Draw positive classified windows
            if (args.display) {
                for (size_t i = 0; i < found->size(); i++) {
                    Rect r = (*found)[i];
                    rectangle(*img_to_show, r.tl(), r.br(), Scalar(0, 255, 0), 3);
                    fprintf(stdout, "point: %d, %d, %d, %d\n", r.tl().x, r.tl().y, r.br().x, r.br().y);
                }

                if (use_gpu)
                    putText(*img_to_show, "Mode: GPU", Point(5, 25), FONT_HERSHEY_SIMPLEX, 1., Scalar(255, 100, 0), 2);
                else
                    putText(*img_to_show, "Mode: CPU", Point(5, 25), FONT_HERSHEY_SIMPLEX, 1., Scalar(255, 100, 0), 2);
                putText(*img_to_show, "FPS HOG: " + hogWorkFps(), Point(5, 65), FONT_HERSHEY_SIMPLEX, 1., Scalar(255, 100, 0), 2);
                putText(*img_to_show, "FPS total: " + workFps(), Point(5, 105), FONT_HERSHEY_SIMPLEX, 1., Scalar(255, 100, 0), 2);
                imshow("opencv_gpu_hog", *img_to_show);
                handleKey((char)waitKey(3));
            }

            workEnd();

            if (args.write_video) {
                if (!video_writer.isOpened()) {
                    video_writer.open(args.dst_video, VideoWriter::fourcc('x','v','i','d'), args.dst_video_fps,
                            img_to_show->size(), true);
                    if (!video_writer.isOpened())
                        throw std::runtime_error("can't create video writer");
                }

                if (make_gray) cvtColor(*img_to_show, *img, COLOR_GRAY2BGR);
                else cvtColor(*img_to_show, *img, COLOR_BGRA2BGR);

                video_writer << *img;
            }

            delete gpu_img;
            delete found;
            delete img;
            gpu_img = new cuda::GpuMat();
            found = new vector<Rect>();
            img = new Mat();
            count_frame++;
            /* Wait until the next job is released. */
            if (args.realtime)
                sleep_next_period();
        }
    }
    /*****
     * 6) Transition to background mode.
     */
    if (args.realtime)
        CALL( task_mode(BACKGROUND_TASK) );
}

void App::sched_etoe_hog(cv::Ptr<cv::cuda::HOG> gpu_hog, cv::HOGDescriptor cpu_hog)
{
    Size win_stride(args.win_stride_width, args.win_stride_height);
    Size win_size(args.win_width, args.win_width * 2);

    Mat img_aux;
    Mat* img = new Mat();
    Mat* img_to_show;
    cuda::GpuMat* gpu_img = new cuda::GpuMat();
    vector<Rect>* found = new vector<Rect>();

    VideoCapture vc;
    Mat frame;
    vector<String> filenames;

    int count_frame = 0;
    while (count_frame < args.count && running)
    {

        unsigned int count = 1;

        if (args.src_is_video)
        {
            vc.open(args.src.c_str());
            if (!vc.isOpened())
                throw runtime_error(string("can't open video file: " + args.src));
            vc >> frame;
        }
        else if (args.src_is_folder) {
            String folder = args.src;
            cout << folder << endl;
            glob(folder, filenames);
            frame = imread(filenames[count]);	// 0 --> .gitignore
            if (!frame.data)
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
            vc >> frame;
        }
        else
        {
            frame = imread(args.src);
            if (frame.empty())
                throw runtime_error(string("can't open image file: " + args.src));
        }

        // Iterate over all frames
        while (count_frame < args.count && running && !frame.empty())
        {
            //usleep(33000);
            workBegin();

            // Change format of the image
            if (make_gray) cvtColor(frame, img_aux, COLOR_BGR2GRAY);
            else if (use_gpu) cvtColor(frame, img_aux, COLOR_BGR2BGRA);
            else frame.copyTo(img_aux);

            // Resize image
            if (args.resize_src) resize(img_aux, *img, Size(args.width, args.height));
            else *img = img_aux;
            img_to_show = img;


            // Perform HOG classification
            hogWorkBegin();
            if (use_gpu)
            {
                gpu_img->upload(*img);
                gpu_hog->setNumLevels(nlevels);
                gpu_hog->setHitThreshold(hit_threshold);
                gpu_hog->setScaleFactor(scale);
                gpu_hog->setGroupThreshold(gr_threshold);
                gpu_hog->detectMultiScale(*gpu_img, *found);
            }
            else
            {
                cpu_hog.nlevels = nlevels;
                cpu_hog.detectMultiScale(*img, *found, hit_threshold, win_stride,
                                         Size(0, 0), scale, gr_threshold);
            }
            hogWorkEnd();

            // Draw positive classified windows
            printf("response time: %f\n", 1/hog_work_fps);
            for (size_t i = 0; i < found->size(); i++)
            {
                Rect r = (*found)[i];
                rectangle(*img_to_show, r.tl(), r.br(), Scalar(0, 255, 0), 3);
            }

            if (use_gpu)
                putText(*img_to_show, "Mode: GPU", Point(5, 25), FONT_HERSHEY_SIMPLEX, 1., Scalar(255, 100, 0), 2);
            else
                putText(*img_to_show, "Mode: CPU", Point(5, 25), FONT_HERSHEY_SIMPLEX, 1., Scalar(255, 100, 0), 2);
            putText(*img_to_show, "FPS HOG: " + hogWorkFps(), Point(5, 65), FONT_HERSHEY_SIMPLEX, 1., Scalar(255, 100, 0), 2);
            putText(*img_to_show, "FPS total: " + workFps(), Point(5, 105), FONT_HERSHEY_SIMPLEX, 1., Scalar(255, 100, 0), 2);
            imshow("opencv_gpu_hog", *img_to_show);

            if (args.src_is_video || args.src_is_camera) vc >> frame;
            if (args.src_is_folder) {
                count++;
                if (count < filenames.size()) {
                    frame = imread(filenames[count]);
                } else {
                    Mat empty;
                    frame = empty;
                }
            }

            workEnd();

            if (args.write_video)
            {
                if (!video_writer.isOpened())
                {
                    video_writer.open(args.dst_video, VideoWriter::fourcc('x','v','i','d'), args.dst_video_fps,
                                      img_to_show->size(), true);
                    if (!video_writer.isOpened())
                        throw std::runtime_error("can't create video writer");
                }

                if (make_gray) cvtColor(*img_to_show, *img, COLOR_GRAY2BGR);
                else cvtColor(*img_to_show, *img, COLOR_BGRA2BGR);

                video_writer << *img;
            }

            handleKey((char)waitKey(3));
            delete gpu_img;
            delete found;
            delete img;
            gpu_img = new cuda::GpuMat();
            found = new vector<Rect>();
            img = new Mat();
            count_frame++;
        }
    }
}

void App::run()
{
    mlockall(MCL_CURRENT | MCL_FUTURE);
    setNumThreads(0);
    running = true;

    Size win_stride(args.win_stride_width, args.win_stride_height);
    Size win_size(args.win_width, args.win_width * 2);
    Size block_size(args.block_width, args.block_width);
    Size block_stride(args.block_stride_width, args.block_stride_height);
    Size cell_size(args.cell_width, args.cell_width);

    cv::Ptr<cv::cuda::HOG> gpu_hog = cv::cuda::HOG::create(win_size, block_size, block_stride, cell_size, args.nbins);
    cv::HOGDescriptor cpu_hog(win_size, block_size, block_stride, cell_size, args.nbins);

    if(args.svm_load) {
        std::vector<float> svm_model;
        const std::string model_file_name = args.svm;
        FileStorage ifs(model_file_name, FileStorage::READ);
        if (ifs.isOpened()) {
            ifs["svm_detector"] >> svm_model;
        } else {
            const std::string what =
                    "could not load model for hog classifier from file: "
                    + model_file_name;
            throw std::runtime_error(what);
        }

        // check if the variables are initialized
        if (svm_model.empty()) {
            const std::string what =
                    "HoG classifier: svm model could not be loaded from file"
                    + model_file_name;
            throw std::runtime_error(what);
        }

        gpu_hog->setSVMDetector(svm_model);
        cpu_hog.setSVMDetector(svm_model);
    } else {
        // Create HOG descriptors and detectors here
        Mat detector = gpu_hog->getDefaultPeopleDetector();

        gpu_hog->setSVMDetector(detector);
        cpu_hog.setSVMDetector(detector);
    }

    cout << "gpusvmDescriptorSize : " << gpu_hog->getDescriptorSize()
         << endl;
    cout << "cpusvmDescriptorSize : " << cpu_hog.getDescriptorSize()
         << endl;

    VideoCapture vc;
    Mat frames[100];
    vector<String> filenames;
    if (args.src_is_video)
    {
        vc.open(args.src.c_str());
        if (!vc.isOpened())
            throw runtime_error(string("can't open video file: " + args.src));
        vc >> frames[0];
    }
    for (int i=1; i<100; i++) {
        if (args.src_is_video) vc >> frames[i];
    }
    vc.release();

    switch (args.sched) {
        case end_to_end:
            sched_etoe_hog_preload(gpu_hog, cpu_hog, frames);
            break;
        case fine_grained:
            sched_fine_grained_hog(gpu_hog, cpu_hog, frames);
            break;
        case coarse_grained:
        case coarse_unrolled:
            sched_coarse_grained_unrolled_for_hog(gpu_hog, cpu_hog, frames);
            break;
        case fine_AB:
            sched_AB_hog(gpu_hog, cpu_hog, frames);
            break;
        case fine_BC:
            sched_BC_hog(gpu_hog, cpu_hog, frames);
            break;
        case fine_CD:
            sched_CD_hog(gpu_hog, cpu_hog, frames);
            break;
        case fine_DE:
            sched_DE_hog(gpu_hog, cpu_hog, frames);
            break;
        case fine_ABC:
            sched_ABC_hog(gpu_hog, cpu_hog, frames);
            break;
        case fine_BCD:
            sched_BCD_hog(gpu_hog, cpu_hog, frames);
            break;
        case fine_CDE:
            sched_CDE_hog(gpu_hog, cpu_hog, frames);
            break;
        case fine_ABCD:
            sched_ABCD_hog(gpu_hog, cpu_hog, frames);
            break;
        case fine_BCDE:
            sched_BCDE_hog(gpu_hog, cpu_hog, frames);
            break;
        case fine_ABCDE:
            sched_ABCDE_hog(gpu_hog, cpu_hog, frames);
            break;
        case fine_single_merge_in_level:
            sched_single_merge_in_level_hog(gpu_hog, cpu_hog, frames);
            break;
        default:
            break;
    }
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
    case 'g':
    case 'G':
        make_gray = !make_gray;
        cout << "Convert image to gray: " << (make_gray ? "YES" : "NO") << endl;
        break;
    case '1':
        scale *= 1.05;
        cout << "Scale: " << scale << endl;
        break;
    case 'q':
    case 'Q':
        scale /= 1.05;
        cout << "Scale: " << scale << endl;
        break;
    case '2':
        nlevels++;
        cout << "Levels number: " << nlevels << endl;
        break;
    case 'w':
    case 'W':
        nlevels = max(nlevels - 1, 1);
        cout << "Levels number: " << nlevels << endl;
        break;
    case '3':
        gr_threshold++;
        cout << "Group threshold: " << gr_threshold << endl;
        break;
    case 'e':
    case 'E':
        gr_threshold = max(0, gr_threshold - 1);
        cout << "Group threshold: " << gr_threshold << endl;
        break;
    case '4':
        hit_threshold+=0.25;
        cout << "Hit threshold: " << hit_threshold << endl;
        break;
    case 'r':
    case 'R':
        hit_threshold = max(0.0, hit_threshold - 0.25);
        cout << "Hit threshold: " << hit_threshold << endl;
        break;
    case 'c':
    case 'C':
        gamma_corr = !gamma_corr;
        cout << "Gamma correction: " << gamma_corr << endl;
        break;
    }
}


inline void App::hogWorkBegin() { hog_work_begin = getTickCount(); }

inline void App::hogWorkEnd()
{
    hog_work_end = getTickCount();

    int64 delta = hog_work_end - hog_work_begin;
    double freq = getTickFrequency();
    hog_work_fps = freq / delta;
}

inline string App::hogWorkFps() const
{
    stringstream ss;
    ss << hog_work_fps;
    return ss.str();
}

inline string App::frameIndex(int i) const
{
    stringstream ss;
    ss << i;
    return ss.str();
}


inline void App::workBegin() { work_begin = getTickCount(); }

inline void App::workEnd()
{
    int64 delta = getTickCount() - work_begin;
    double freq = getTickFrequency();
    work_fps = freq / delta;
}

inline string App::workFps() const
{
    stringstream ss;
    ss << work_fps;
    return ss.str();
}


void App::sched_AB_hog(cv::Ptr<cv::cuda::HOG> gpu_hog, cv::HOGDescriptor cpu_hog, Mat* frames)
{
    pthread_barrier_t arr_fine_init_barrier[args.num_fine_graphs];
    /* graph construction */
    graph_t arr_g [args.num_fine_graphs];

    struct sync_info arr_sync_info_AB                 [args.num_fine_graphs][NUM_SCALE_LEVELS];
    struct sync_info arr_sync_info_compute_histograms [args.num_fine_graphs][NUM_SCALE_LEVELS];
    struct sync_info arr_sync_info_normalize          [args.num_fine_graphs][NUM_SCALE_LEVELS];
    struct sync_info arr_sync_info_classify           [args.num_fine_graphs][NUM_SCALE_LEVELS];

    thread** arr_t0  = (thread**) calloc(args.num_fine_graphs, sizeof(std::thread *));
    thread** arr_t1  = (thread**) calloc(args.num_fine_graphs, sizeof(std::thread *));
    thread** arr_t23 = (thread**) calloc(args.num_fine_graphs * NUM_SCALE_LEVELS, sizeof(std::thread *));
    thread** arr_t4  = (thread**) calloc(args.num_fine_graphs * NUM_SCALE_LEVELS, sizeof(std::thread *));
    thread** arr_t5  = (thread**) calloc(args.num_fine_graphs * NUM_SCALE_LEVELS, sizeof(std::thread *));
    thread** arr_t6  = (thread**) calloc(args.num_fine_graphs * NUM_SCALE_LEVELS, sizeof(std::thread *));
    thread** arr_t7  = (thread**) calloc(args.num_fine_graphs, sizeof(std::thread *));
    thread** arr_t8  = (thread**) calloc(args.num_fine_graphs, sizeof(std::thread *));

    char buf[30];
    sprintf(buf, "/tmp/graph_t%d", args.task_id);
    CheckError(pgm_init(buf, 1));

    for (int g_idx = 0; g_idx < args.num_fine_graphs; g_idx++)
    {
        pthread_barrier_t* fine_init_barrier = arr_fine_init_barrier + g_idx;

        // A graph consists of the graph itself, nodes, and edge
        graph_t* g_ptr = arr_g + g_idx;

        node_t color_convert_node;
        node_t compute_scales_node;
        node_t AB_node                 [NUM_SCALE_LEVELS]; // resize + compute-grads
        node_t compute_histograms_node [NUM_SCALE_LEVELS];
        node_t normalize_node          [NUM_SCALE_LEVELS];
        node_t classify_node           [NUM_SCALE_LEVELS];
        node_t collect_locations_node;
        node_t display_node;

        edge_t e0_1;
        edge_t e1_23 [NUM_SCALE_LEVELS];
        edge_t e23_4 [NUM_SCALE_LEVELS];
        edge_t e4_5  [NUM_SCALE_LEVELS];
        edge_t e5_6  [NUM_SCALE_LEVELS];
        edge_t e6_7  [NUM_SCALE_LEVELS];
        edge_t e7_8;

        // Initialize the graph
        sprintf(buf, "hog_%d", g_idx);
        CheckError(pgm_init_graph(g_ptr, buf));
        graph_t g = *g_ptr;

        // Initialize the nodes
        CheckError(pgm_init_node(&color_convert_node, g, "color_convert"));
        CheckError(pgm_init_node(&compute_scales_node, g, "compute_scales"));
        for (int i=0; i<NUM_SCALE_LEVELS; i++) {
            CheckError(pgm_init_node(AB_node + i, g, "AB"));
            CheckError(pgm_init_node(compute_histograms_node + i, g, "compute_histograms"));
            CheckError(pgm_init_node(normalize_node + i, g, "normalize"));
            CheckError(pgm_init_node(classify_node + i, g, "classify"));
        }
        CheckError(pgm_init_node(&collect_locations_node, g, "collect_locations"));
        CheckError(pgm_init_node(&display_node, g, "display"));

        // Initialize the edges
        edge_attr_t fast_mq_attr;
        memset(&fast_mq_attr, 0, sizeof(fast_mq_attr));
        fast_mq_attr.type = pgm_fast_fifo_edge;

        fast_mq_attr.nr_produce = sizeof(struct params_compute);
        fast_mq_attr.nr_consume = sizeof(struct params_compute);
        fast_mq_attr.nr_threshold = sizeof(struct params_compute);
        CheckError(pgm_init_edge(&e0_1, color_convert_node, compute_scales_node, "e0_1", &fast_mq_attr));

        for (int i=0; i<NUM_SCALE_LEVELS; i++) {
            sprintf(buf, "e1_23_%d", i);
            fast_mq_attr.nr_produce = sizeof(struct params_resize);
            fast_mq_attr.nr_consume = sizeof(struct params_resize);
            fast_mq_attr.nr_threshold = sizeof(struct params_resize);
            CheckError(pgm_init_edge(e1_23 + i,
                        compute_scales_node,
                        AB_node[i], buf, &fast_mq_attr));

            sprintf(buf, "e23_4_%d", i);
            fast_mq_attr.nr_produce = sizeof(struct params_compute_histograms);
            fast_mq_attr.nr_consume = sizeof(struct params_compute_histograms);
            fast_mq_attr.nr_threshold = sizeof(struct params_compute_histograms);
            CheckError(pgm_init_edge(e23_4 + i,
                        AB_node[i],
                        compute_histograms_node[i], buf, &fast_mq_attr));

            sprintf(buf, "e4_5_%d", i);
            fast_mq_attr.nr_produce = sizeof(struct params_fine_normalize);
            fast_mq_attr.nr_consume = sizeof(struct params_fine_normalize);
            fast_mq_attr.nr_threshold = sizeof(struct params_fine_normalize);
            CheckError(pgm_init_edge(e4_5 + i,
                        compute_histograms_node[i],
                        normalize_node[i], buf, &fast_mq_attr));

            sprintf(buf, "e5_6_%d", i);
            fast_mq_attr.nr_produce = sizeof(struct params_fine_classify);
            fast_mq_attr.nr_consume = sizeof(struct params_fine_classify);
            fast_mq_attr.nr_threshold = sizeof(struct params_fine_classify);
            CheckError(pgm_init_edge(e5_6 + i,
                        normalize_node[i],
                        classify_node[i], buf, &fast_mq_attr));

            sprintf(buf, "e6_7_%d", i);
            fast_mq_attr.nr_produce = sizeof(struct params_fine_collect_locations);
            fast_mq_attr.nr_consume = sizeof(struct params_fine_collect_locations);
            fast_mq_attr.nr_threshold = sizeof(struct params_fine_collect_locations);
            CheckError(pgm_init_edge(e6_7 + i,
                        classify_node[i],
                        collect_locations_node, buf, &fast_mq_attr));
        }

        fast_mq_attr.nr_produce = sizeof(struct params_display);
        fast_mq_attr.nr_consume = sizeof(struct params_display);
        fast_mq_attr.nr_threshold = sizeof(struct params_display);
        CheckError(pgm_init_edge(&e7_8, collect_locations_node, display_node, "e7_8", &fast_mq_attr));

        pthread_barrier_init(fine_init_barrier, 0, 4 * NUM_SCALE_LEVELS + 4);

        thread** t0  = arr_t0 + g_idx;
        thread** t1  = arr_t1 + g_idx;
        thread** t23 = arr_t23 + g_idx * NUM_SCALE_LEVELS;
        thread** t4  = arr_t4 + g_idx * NUM_SCALE_LEVELS;
        thread** t5  = arr_t5 + g_idx * NUM_SCALE_LEVELS;
        thread** t6  = arr_t6 + g_idx * NUM_SCALE_LEVELS;
        thread** t7  = arr_t7 + g_idx;
        thread** t8  = arr_t8 + g_idx;

        struct sync_info* in_sync_info_AB                 = arr_sync_info_AB                [((g_idx + args.num_fine_graphs - 1) % args.num_fine_graphs)];
        struct sync_info* in_sync_info_compute_histograms = arr_sync_info_compute_histograms[((g_idx + args.num_fine_graphs - 1) % args.num_fine_graphs)];
        struct sync_info* in_sync_info_normalize          = arr_sync_info_normalize         [((g_idx + args.num_fine_graphs - 1) % args.num_fine_graphs)];
        struct sync_info* in_sync_info_classify           = arr_sync_info_classify          [((g_idx + args.num_fine_graphs - 1) % args.num_fine_graphs)];

        struct sync_info* out_sync_info_AB                 = arr_sync_info_AB                [g_idx];
        struct sync_info* out_sync_info_compute_histograms = arr_sync_info_compute_histograms[g_idx];
        struct sync_info* out_sync_info_normalize          = arr_sync_info_normalize         [g_idx];
        struct sync_info* out_sync_info_classify           = arr_sync_info_classify          [g_idx];

        for (int i=0; i<NUM_SCALE_LEVELS; i++) {
            sync_info_init(i + in_sync_info_AB);
            sync_info_init(i + in_sync_info_compute_histograms);
            sync_info_init(i + in_sync_info_normalize);
            sync_info_init(i + in_sync_info_classify);

            sync_info_init(i + out_sync_info_AB);
            sync_info_init(i + out_sync_info_compute_histograms);
            sync_info_init(i + out_sync_info_normalize);
            sync_info_init(i + out_sync_info_classify);
        }

        /* G-FL fine-grained HOG */
        float bound_color_convert                   = 31.0194960215;
        float bound_compute_scales                  = 30.5688293615;
        float bounds_AB          [NUM_SCALE_LEVELS] = {30.2419972231,30.3148010256,30.2936775569,30.3079650324,30.2963991483,30.3058228698,30.3005336032,30.2888197835,30.293491626,30.2840584057,30.3013304497,30.3016323015,30.3111116087};
        float bounds_compute_hist[NUM_SCALE_LEVELS] = {30.5326861275,30.4951636278,30.5415047381,30.4958614402,30.5149987735,30.4750125581,30.5373623676,30.522424222,30.4851937235,30.4577396346,30.4263855713,30.4118414514,30.5202006157};
        float bounds_normalize   [NUM_SCALE_LEVELS] = {30.2903383654,30.3080869341,30.2928467608,30.3064130287,30.3114078316,30.3120987837,30.2985990076,30.2803770804,30.3099685825,30.3123524377,30.2989110618,30.2943410679,30.3081075149};
        float bounds_classify    [NUM_SCALE_LEVELS] = {30.4792191337,30.4948973087,30.4540269983,30.429525815,30.5229441951,30.5070612675,30.497249677,30.4955912512,30.4267037822,30.4508744413,30.430038752,30.3991297266,30.3767439685};
        //float bound_collect_locations               30.983976896;

        /* G-FL fine-grained HOG */
        float cost_color_convert                   = 4.509346;
        float cost_compute_scales                  = 1.947347;
        float costs_AB          [NUM_SCALE_LEVELS] = {0.089336,0.503219,0.383134,0.464357,0.398606,0.452179,0.42211,0.355518,0.382077,0.32845,0.42664,0.428356,0.482245};
        float costs_compute_hist[NUM_SCALE_LEVELS] = {1.741876,1.528564,1.792009,1.532531,1.641325,1.414007,1.76846,1.683538,1.471886,1.315812,1.137567,1.054885,1.670897};
        float costs_normalize   [NUM_SCALE_LEVELS] = {0.364151,0.46505,0.378411,0.455534,0.483929,0.487857,0.411112,0.307522,0.475747,0.489299,0.412886,0.386906,0.465167};
        float costs_classify    [NUM_SCALE_LEVELS] = {1.437921,1.52705,1.294706,1.155419,1.686494,1.596201,1.540423,1.530995,1.139376,1.276784,1.158335,0.98262,0.855359};
        float cost_collect_locations               = 4.307423;

        /* C-FL fine-grained HOG */
        //float bound_color_convert                   = 26.3496352098;
        //float bound_compute_scales                  = 25.6841982494;
        //float bounds_AB          [NUM_SCALE_LEVELS] = {25.0939476981,25.1387935171,25.130189875,25.1357033275,25.1312621102,25.1463332249,25.1316714257,25.1302245094,25.1267072576,25.1262401226,25.1348909935,25.1432006726,25.1413123823};
        //float bounds_compute_hist[NUM_SCALE_LEVELS] = {25.5728014354,25.4973631396,25.5123077387,25.471628638,25.3841716342,25.3274278981,25.4720253593,25.5328196023,25.4654399579,25.3426074958,25.4860697502,25.3209827517,25.4610373822};
        //float bounds_normalize   [NUM_SCALE_LEVELS] = {25.1658996554,25.1786894798,25.1487118351,25.1680595823,25.1750299693,25.1885187788,25.1646279147,25.1649092834,25.1582577618,25.1765865135,25.1940356661,25.1748711091,25.179600851};
        //float bounds_classify    [NUM_SCALE_LEVELS] = {25.2909527262,25.3017792668,25.4537335318,25.3216511097,25.4693284561,25.3272781973,25.3334016159,25.3265829333,25.3061011243,25.2757711248,25.2811148401,25.2748319888,25.3276382806};
        ////float bound_collect_locations               26.204139258;

        /////* C-FL fine-grained HOG */
        //float cost_color_convert                   = 4.475663;
        //float cost_compute_scales                  = 2.150868;
        //float costs_AB          [NUM_SCALE_LEVELS] = {0.088747,0.245422,0.215364,0.234626,0.21911,0.271763,0.22054,0.215485,0.203197,0.201565,0.231788,0.260819,0.254222};
        //float costs_compute_hist[NUM_SCALE_LEVELS] = {1.761688,1.498134,1.550345,1.408227,1.102684,0.904442,1.409613,1.622006,1.386606,0.957474,1.458679,0.881925,1.371225};
        //float costs_normalize   [NUM_SCALE_LEVELS] = {0.340121,0.384804,0.280073,0.347667,0.372019,0.419144,0.335678,0.336661,0.313423,0.377457,0.438418,0.371464,0.387988};
        //float costs_classify    [NUM_SCALE_LEVELS] = {0.777011,0.814835,1.345708,0.88426,1.400191,0.903919,0.925312,0.90149,0.829934,0.723972,0.742641,0.720691,0.905177};
        //float cost_collect_locations               = 3.967353;

        /* | first graph release      | second graph release     | first graph release again
         *  <---------PERIOD--------->
         *  <--------------- PERIOD * args.num_fine_graphs ---------->
         */
        int period = PERIOD * args.num_fine_graphs;
        int m_cpus = 16;
        struct task_info t_info;
        t_info.early = args.early;
        t_info.realtime = args.realtime;
        t_info.sched = fine_AB;
        t_info.period = period;
        t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, cost_color_convert);
        t_info.phase = PERIOD * g_idx;
        t_info.id = 0;
        if (args.cluster != -1)
            t_info.cluster = args.cluster;
        else
            t_info.cluster = args.cluster;
        *t0 = new thread(&App::thread_color_convert, this,
                &color_convert_node, fine_init_barrier,
                gpu_hog, cpu_hog, frames, t_info, g_idx);

        t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, cost_compute_scales);
        t_info.id = 1;
        t_info.phase = t_info.phase + bound_color_convert;
        *t1 = new thread(&cv::cuda::HOG::thread_fine_compute_scales, gpu_hog,
                &compute_scales_node, fine_init_barrier, t_info);

        for (int i=0; i<NUM_SCALE_LEVELS; i++) {
            t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, costs_AB[i]);
            t_info.id = 0 * NUM_SCALE_LEVELS + i + 2;
            t_info.phase = PERIOD * g_idx + bound_color_convert + bound_compute_scales;
            t_info.s_info_in = in_sync_info_AB + i;
            t_info.s_info_out = out_sync_info_AB + i;
            t23[i] = new thread(&cv::cuda::HOG::thread_fine_AB, gpu_hog,
                    AB_node + i, fine_init_barrier, t_info);

            t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, costs_compute_hist[i]);
            t_info.id = 1 * NUM_SCALE_LEVELS + i + 2;
            t_info.phase = t_info.phase + bounds_AB[i];
            t_info.s_info_in = in_sync_info_compute_histograms + i;
            t_info.s_info_out = out_sync_info_compute_histograms + i;
            t4[i] = new thread(&cv::cuda::HOG::thread_fine_compute_histograms,
                    gpu_hog, compute_histograms_node + i, fine_init_barrier,
                    t_info);

            t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, costs_normalize[i]);
            t_info.id = 2 * NUM_SCALE_LEVELS + i + 2;
            t_info.phase = t_info.phase + bounds_compute_hist[i];
            t_info.s_info_in = in_sync_info_normalize + i;
            t_info.s_info_out = out_sync_info_normalize + i;
            t5[i] = new thread(&cv::cuda::HOG::thread_fine_normalize_histograms,
                    gpu_hog, normalize_node + i, fine_init_barrier, t_info);

            t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, costs_classify[i]);
            t_info.id = 3 * NUM_SCALE_LEVELS + i + 2;
            t_info.phase = t_info.phase + bounds_normalize[i];
            t_info.s_info_in = in_sync_info_classify + i;
            t_info.s_info_out = out_sync_info_classify + i;
            t6[i] = new thread(&cv::cuda::HOG::thread_fine_classify, gpu_hog,
                    classify_node + i, fine_init_barrier, t_info);
        }

        t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, cost_collect_locations);
        t_info.id = 4 * NUM_SCALE_LEVELS + 2;
        t_info.phase = t_info.phase + *std::max_element(bounds_classify, bounds_classify+NUM_SCALE_LEVELS);
        *t7 = new thread(&cv::cuda::HOG::thread_fine_collect_locations, gpu_hog,
                &collect_locations_node, fine_init_barrier, t_info);
        //t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, cost_display);
        //t_info.phase;
        *t8 = new thread(&App::thread_display, this, &display_node,
                fine_init_barrier, g_idx == 0 && args.display);
    }

    /* graph construction finishes */

    printf("Joining pthreads...\n");

    for (int g_idx = 0; g_idx < args.num_fine_graphs; g_idx++) {
        graph_t g = arr_g[g_idx];
        thread* t0 = arr_t0[g_idx];
        thread* t1 = arr_t1[g_idx];
        thread** t23 = arr_t23 + g_idx * NUM_SCALE_LEVELS;
        thread** t4  = arr_t4 + g_idx * NUM_SCALE_LEVELS;
        thread** t5  = arr_t5 + g_idx * NUM_SCALE_LEVELS;
        thread** t6  = arr_t6 + g_idx * NUM_SCALE_LEVELS;
        thread* t7 = arr_t7[g_idx];
        thread* t8 = arr_t8[g_idx];
        t0->join();
        t1->join();
        delete t0;
        delete t1;
        for (int i=0; i<NUM_SCALE_LEVELS; i++) {
            if (t23[i]->joinable()) t23[i]->join();
            if (t4[i]->joinable()) t4[i]->join();
            if (t5[i]->joinable()) t5[i]->join();
            if (t6[i]->joinable()) t6[i]->join();

            delete t23[i];
            delete t4[i];
            delete t5[i];
            delete t6[i];
        }
        t7->join();
        t8->join();
        delete t7;
        delete t8;
        CheckError(pgm_destroy_graph(g));
    }
    free(arr_t1);
    free(arr_t23);
    free(arr_t4);
    free(arr_t5);
    free(arr_t6);
    free(arr_t7);
    free(arr_t8);

    //CheckError(pgm_destroy_graph(g));
    CheckError(pgm_destroy());
    fprintf(stdout, "cleaned up ...");
}

void App::sched_BC_hog(cv::Ptr<cv::cuda::HOG> gpu_hog, cv::HOGDescriptor cpu_hog, Mat* frames)
{
    pthread_barrier_t arr_fine_init_barrier[args.num_fine_graphs];
    /* graph construction */
    graph_t arr_g [args.num_fine_graphs];

    struct sync_info arr_sync_info_resize    [args.num_fine_graphs][NUM_SCALE_LEVELS];
    struct sync_info arr_sync_info_BC        [args.num_fine_graphs][NUM_SCALE_LEVELS];
    struct sync_info arr_sync_info_normalize [args.num_fine_graphs][NUM_SCALE_LEVELS];
    struct sync_info arr_sync_info_classify  [args.num_fine_graphs][NUM_SCALE_LEVELS];

    thread** arr_t0  = (thread**) calloc(args.num_fine_graphs, sizeof(std::thread *));
    thread** arr_t1  = (thread**) calloc(args.num_fine_graphs, sizeof(std::thread *));
    thread** arr_t2  = (thread**) calloc(args.num_fine_graphs * NUM_SCALE_LEVELS, sizeof(std::thread *));
    thread** arr_t34 = (thread**) calloc(args.num_fine_graphs * NUM_SCALE_LEVELS, sizeof(std::thread *));
    thread** arr_t5  = (thread**) calloc(args.num_fine_graphs * NUM_SCALE_LEVELS, sizeof(std::thread *));
    thread** arr_t6  = (thread**) calloc(args.num_fine_graphs * NUM_SCALE_LEVELS, sizeof(std::thread *));
    thread** arr_t7  = (thread**) calloc(args.num_fine_graphs, sizeof(std::thread *));
    thread** arr_t8  = (thread**) calloc(args.num_fine_graphs, sizeof(std::thread *));

    char buf[30];
    sprintf(buf, "/tmp/graph_t%d", args.task_id);
    CheckError(pgm_init(buf, 1));

    for (int g_idx = 0; g_idx < args.num_fine_graphs; g_idx++)
    {
        pthread_barrier_t* fine_init_barrier = arr_fine_init_barrier + g_idx;

        // A graph consists of the graph itself, nodes, and edge
        graph_t* g_ptr = arr_g + g_idx;

        node_t color_convert_node;
        node_t compute_scales_node;
        node_t resize_node             [NUM_SCALE_LEVELS];
        node_t BC_node                 [NUM_SCALE_LEVELS]; // compute-grads + compute-hists
        node_t normalize_node          [NUM_SCALE_LEVELS];
        node_t classify_node           [NUM_SCALE_LEVELS];
        node_t collect_locations_node;
        node_t display_node;

        edge_t e0_1;
        edge_t e1_2  [NUM_SCALE_LEVELS];
        edge_t e2_34 [NUM_SCALE_LEVELS];
        edge_t e34_5 [NUM_SCALE_LEVELS];
        edge_t e5_6  [NUM_SCALE_LEVELS];
        edge_t e6_7  [NUM_SCALE_LEVELS];
        edge_t e7_8;

        // Initialize the graph
        sprintf(buf, "hog_%d", g_idx);
        CheckError(pgm_init_graph(g_ptr, buf));
        graph_t g = *g_ptr;

        // Initialize the nodes
        CheckError(pgm_init_node(&color_convert_node, g, "color_convert"));
        CheckError(pgm_init_node(&compute_scales_node, g, "compute_scales"));
        for (int i=0; i<NUM_SCALE_LEVELS; i++) {
            CheckError(pgm_init_node(resize_node + i, g, "resize"));
            CheckError(pgm_init_node(BC_node + i, g, "BC"));
            CheckError(pgm_init_node(normalize_node + i, g, "normalize"));
            CheckError(pgm_init_node(classify_node + i, g, "classify"));
        }
        CheckError(pgm_init_node(&collect_locations_node, g, "collect_locations"));
        CheckError(pgm_init_node(&display_node, g, "display"));

        // Initialize the edges
        edge_attr_t fast_mq_attr;
        memset(&fast_mq_attr, 0, sizeof(fast_mq_attr));
        fast_mq_attr.type = pgm_fast_fifo_edge;

        fast_mq_attr.nr_produce = sizeof(struct params_compute);
        fast_mq_attr.nr_consume = sizeof(struct params_compute);
        fast_mq_attr.nr_threshold = sizeof(struct params_compute);
        CheckError(pgm_init_edge(&e0_1, color_convert_node, compute_scales_node, "e0_1", &fast_mq_attr));

        for (int i=0; i<NUM_SCALE_LEVELS; i++) {
            sprintf(buf, "e1_2_%d", i);
            fast_mq_attr.nr_produce = sizeof(struct params_resize);
            fast_mq_attr.nr_consume = sizeof(struct params_resize);
            fast_mq_attr.nr_threshold = sizeof(struct params_resize);
            CheckError(pgm_init_edge(e1_2 + i,
                        compute_scales_node,
                        resize_node[i], buf, &fast_mq_attr));

            sprintf(buf, "e2_34_%d", i);
            fast_mq_attr.nr_produce = sizeof(struct params_compute_gradients);
            fast_mq_attr.nr_consume = sizeof(struct params_compute_gradients);
            fast_mq_attr.nr_threshold = sizeof(struct params_compute_gradients);
            CheckError(pgm_init_edge(e2_34 + i,
                        resize_node[i],
                        BC_node[i], buf, &fast_mq_attr));

            sprintf(buf, "e34_5_%d", i);
            fast_mq_attr.nr_produce = sizeof(struct params_fine_normalize);
            fast_mq_attr.nr_consume = sizeof(struct params_fine_normalize);
            fast_mq_attr.nr_threshold = sizeof(struct params_fine_normalize);
            CheckError(pgm_init_edge(e34_5 + i,
                        BC_node[i],
                        normalize_node[i], buf, &fast_mq_attr));

            sprintf(buf, "e5_6_%d", i);
            fast_mq_attr.nr_produce = sizeof(struct params_fine_classify);
            fast_mq_attr.nr_consume = sizeof(struct params_fine_classify);
            fast_mq_attr.nr_threshold = sizeof(struct params_fine_classify);
            CheckError(pgm_init_edge(e5_6 + i,
                        normalize_node[i],
                        classify_node[i], buf, &fast_mq_attr));

            sprintf(buf, "e6_7_%d", i);
            fast_mq_attr.nr_produce = sizeof(struct params_fine_collect_locations);
            fast_mq_attr.nr_consume = sizeof(struct params_fine_collect_locations);
            fast_mq_attr.nr_threshold = sizeof(struct params_fine_collect_locations);
            CheckError(pgm_init_edge(e6_7 + i,
                        classify_node[i],
                        collect_locations_node, buf, &fast_mq_attr));
        }

        fast_mq_attr.nr_produce = sizeof(struct params_display);
        fast_mq_attr.nr_consume = sizeof(struct params_display);
        fast_mq_attr.nr_threshold = sizeof(struct params_display);
        CheckError(pgm_init_edge(&e7_8, collect_locations_node, display_node, "e7_8", &fast_mq_attr));

        pthread_barrier_init(fine_init_barrier, 0, 4 * NUM_SCALE_LEVELS + 4);

        thread** t0  = arr_t0 + g_idx;
        thread** t1  = arr_t1 + g_idx;
        thread** t2  = arr_t2 + g_idx * NUM_SCALE_LEVELS;
        thread** t34 = arr_t34 + g_idx * NUM_SCALE_LEVELS;
        thread** t5  = arr_t5 + g_idx * NUM_SCALE_LEVELS;
        thread** t6  = arr_t6 + g_idx * NUM_SCALE_LEVELS;
        thread** t7  = arr_t7 + g_idx;
        thread** t8  = arr_t8 + g_idx;

        struct sync_info* in_sync_info_resize             = arr_sync_info_resize            [((g_idx + args.num_fine_graphs - 1) % args.num_fine_graphs)];
        struct sync_info* in_sync_info_BC                 = arr_sync_info_BC                [((g_idx + args.num_fine_graphs - 1) % args.num_fine_graphs)];
        struct sync_info* in_sync_info_normalize          = arr_sync_info_normalize         [((g_idx + args.num_fine_graphs - 1) % args.num_fine_graphs)];
        struct sync_info* in_sync_info_classify           = arr_sync_info_classify          [((g_idx + args.num_fine_graphs - 1) % args.num_fine_graphs)];

        struct sync_info* out_sync_info_resize             = arr_sync_info_resize            [g_idx];
        struct sync_info* out_sync_info_BC                 = arr_sync_info_BC                [g_idx];
        struct sync_info* out_sync_info_normalize          = arr_sync_info_normalize         [g_idx];
        struct sync_info* out_sync_info_classify           = arr_sync_info_classify          [g_idx];

        for (int i=0; i<NUM_SCALE_LEVELS; i++) {
            sync_info_init(i + in_sync_info_resize);
            sync_info_init(i + in_sync_info_BC);
            sync_info_init(i + in_sync_info_normalize);
            sync_info_init(i + in_sync_info_classify);

            sync_info_init(i + out_sync_info_resize);
            sync_info_init(i + out_sync_info_BC);
            sync_info_init(i + out_sync_info_normalize);
            sync_info_init(i + out_sync_info_classify);
        }

        /* G-FL fine-grained HOG */
        float bound_color_convert                   = 31.0194960215;
        float bound_compute_scales                  = 30.5688293615;
        float bounds_resize      [NUM_SCALE_LEVELS] = {30.2419972231,30.3148010256,30.2936775569,30.3079650324,30.2963991483,30.3058228698,30.3005336032,30.2888197835,30.293491626,30.2840584057,30.3013304497,30.3016323015,30.3111116087};
        float bounds_BC          [NUM_SCALE_LEVELS] = {30.4902863291,30.44552396,30.4706995604,30.4291495557,30.4171804989,30.4135009326,30.3785928987,30.4059076714,30.3816881109,30.3627509567,30.3564101342,30.3519205285,30.3405616838};
        float bounds_normalize   [NUM_SCALE_LEVELS] = {30.2903383654,30.3080869341,30.2928467608,30.3064130287,30.3114078316,30.3120987837,30.2985990076,30.2803770804,30.3099685825,30.3123524377,30.2989110618,30.2943410679,30.3081075149};
        float bounds_classify    [NUM_SCALE_LEVELS] = {30.4792191337,30.4948973087,30.4540269983,30.429525815,30.5229441951,30.5070612675,30.497249677,30.4955912512,30.4267037822,30.4508744413,30.430038752,30.3991297266,30.3767439685};
        //float bound_collect_locations               30.983976896;

        /* G-FL fine-grained HOG */
        float cost_color_convert                   = 4.509346;
        float cost_compute_scales                  = 1.947347;
        float costs_resize      [NUM_SCALE_LEVELS] = {0.089336,0.503219,0.383134,0.464357,0.398606,0.452179,0.42211,0.355518,0.382077,0.32845,0.42664,0.428356,0.482245};
        float costs_BC          [NUM_SCALE_LEVELS] = {1.500837,1.246367,1.389488,1.15328,1.085237,1.064319,0.86587,1.021152,0.883466,0.77581,0.739763,0.71424,0.649666};
        float costs_normalize   [NUM_SCALE_LEVELS] = {0.364151,0.46505,0.378411,0.455534,0.483929,0.487857,0.411112,0.307522,0.475747,0.489299,0.412886,0.386906,0.465167};
        float costs_classify    [NUM_SCALE_LEVELS] = {1.437921,1.52705,1.294706,1.155419,1.686494,1.596201,1.540423,1.530995,1.139376,1.276784,1.158335,0.98262,0.855359};
        float cost_collect_locations               = 4.307423;

        /* C-FL fine-grained HOG */
        //float bound_color_convert                   = 26.3496352098;
        //float bound_compute_scales                  = 25.6841982494;
        //float bounds_resize      [NUM_SCALE_LEVELS] = {25.0939476981,25.1387935171,25.130189875,25.1357033275,25.1312621102,25.1463332249,25.1316714257,25.1302245094,25.1267072576,25.1262401226,25.1348909935,25.1432006726,25.1413123823};
        //float bounds_compute_grad[NUM_SCALE_LEVELS] = {25.3489965402,25.4185896374,25.4092134479,25.4079932294,25.4051932818,25.4030496702,25.312479578,25.291207475,25.2868561354,25.2685316772,25.2518075571,25.246440943,25.2364255915};
        //float bounds_BC          [NUM_SCALE_LEVELS] = {25.3489965402,25.4185896374,25.4092134479,25.4079932294,25.4051932818,25.4030496702,25.312479578,25.291207475,25.2868561354,25.2685316772,25.2518075571,25.246440943,25.2364255915};
        //float bounds_compute_hist[NUM_SCALE_LEVELS] = {25.5728014354,25.4973631396,25.5123077387,25.471628638,25.3841716342,25.3274278981,25.4720253593,25.5328196023,25.4654399579,25.3426074958,25.4860697502,25.3209827517,25.4610373822};
        //float bounds_normalize   [NUM_SCALE_LEVELS] = {25.1658996554,25.1786894798,25.1487118351,25.1680595823,25.1750299693,25.1885187788,25.1646279147,25.1649092834,25.1582577618,25.1765865135,25.1940356661,25.1748711091,25.179600851};
        //float bounds_classify    [NUM_SCALE_LEVELS] = {25.2909527262,25.3017792668,25.4537335318,25.3216511097,25.4693284561,25.3272781973,25.3334016159,25.3265829333,25.3061011243,25.2757711248,25.2811148401,25.2748319888,25.3276382806};
        ////float bound_collect_locations               26.204139258;

        /////* C-FL fine-grained HOG */
        //float cost_color_convert                   = 4.475663;
        //float cost_compute_scales                  = 2.150868;
        //float costs_resize      [NUM_SCALE_LEVELS] = {0.088747,0.245422,0.215364,0.234626,0.21911,0.271763,0.22054,0.215485,0.203197,0.201565,0.231788,0.260819,0.254222};
        //float costs_compute_grad[NUM_SCALE_LEVELS] = {0.979795,1.222928,1.190171,1.185908,1.176126,1.168637,0.852218,0.777901,0.762699,0.69868,0.640252,0.621503,0.586513};
        //float costs_BC          [NUM_SCALE_LEVELS] = {0.979795,1.222928,1.190171,1.185908,1.176126,1.168637,0.852218,0.777901,0.762699,0.69868,0.640252,0.621503,0.586513};
        //float costs_compute_hist[NUM_SCALE_LEVELS] = {1.761688,1.498134,1.550345,1.408227,1.102684,0.904442,1.409613,1.622006,1.386606,0.957474,1.458679,0.881925,1.371225};
        //float costs_normalize   [NUM_SCALE_LEVELS] = {0.340121,0.384804,0.280073,0.347667,0.372019,0.419144,0.335678,0.336661,0.313423,0.377457,0.438418,0.371464,0.387988};
        //float costs_classify    [NUM_SCALE_LEVELS] = {0.777011,0.814835,1.345708,0.88426,1.400191,0.903919,0.925312,0.90149,0.829934,0.723972,0.742641,0.720691,0.905177};
        //float cost_collect_locations               = 3.967353;

        /* | first graph release      | second graph release     | first graph release again
         *  <---------PERIOD--------->
         *  <--------------- PERIOD * args.num_fine_graphs ---------->
         */
        int period = PERIOD * args.num_fine_graphs;
        int m_cpus = 16;
        struct task_info t_info;
        t_info.early = args.early;
        t_info.realtime = args.realtime;
        t_info.sched = fine_BC;
        t_info.period = period;
        t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, cost_color_convert);
        t_info.phase = PERIOD * g_idx;
        t_info.id = 0;
        if (args.cluster != -1)
            t_info.cluster = args.cluster;
        else
            t_info.cluster = args.cluster;
        *t0 = new thread(&App::thread_color_convert, this,
                &color_convert_node, fine_init_barrier,
                gpu_hog, cpu_hog, frames, t_info, g_idx);

        t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, cost_compute_scales);
        t_info.id = 1;
        t_info.phase = t_info.phase + bound_color_convert;
        *t1 = new thread(&cv::cuda::HOG::thread_fine_compute_scales, gpu_hog,
                &compute_scales_node, fine_init_barrier, t_info);

        for (int i=0; i<NUM_SCALE_LEVELS; i++) {
            t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, costs_resize[i]);
            t_info.id = 0 * NUM_SCALE_LEVELS + i + 2;
            t_info.phase = PERIOD * g_idx + bound_color_convert + bound_compute_scales;
            t_info.s_info_in = in_sync_info_resize + i;
            t_info.s_info_out = out_sync_info_resize + i;
            t2[i] = new thread(&cv::cuda::HOG::thread_fine_resize, gpu_hog,
                    resize_node + i, fine_init_barrier, t_info);

            t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, costs_BC[i]);
            t_info.id = 1 * NUM_SCALE_LEVELS + i + 2;
            t_info.phase = PERIOD * g_idx + bounds_resize[i];
            t_info.s_info_in = in_sync_info_BC + i;
            t_info.s_info_out = out_sync_info_BC + i;
            t34[i] = new thread(&cv::cuda::HOG::thread_fine_BC, gpu_hog,
                    BC_node + i, fine_init_barrier, t_info);

            t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, costs_normalize[i]);
            t_info.id = 2 * NUM_SCALE_LEVELS + i + 2;
            t_info.phase = t_info.phase + bounds_BC[i];
            t_info.s_info_in = in_sync_info_normalize + i;
            t_info.s_info_out = out_sync_info_normalize + i;
            t5[i] = new thread(&cv::cuda::HOG::thread_fine_normalize_histograms,
                    gpu_hog, normalize_node + i, fine_init_barrier, t_info);

            t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, costs_classify[i]);
            t_info.id = 3 * NUM_SCALE_LEVELS + i + 2;
            t_info.phase = t_info.phase + bounds_normalize[i];
            t_info.s_info_in = in_sync_info_classify + i;
            t_info.s_info_out = out_sync_info_classify + i;
            t6[i] = new thread(&cv::cuda::HOG::thread_fine_classify, gpu_hog,
                    classify_node + i, fine_init_barrier, t_info);
        }

        t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, cost_collect_locations);
        t_info.id = 4 * NUM_SCALE_LEVELS + 2;
        t_info.phase = t_info.phase + *std::max_element(bounds_classify, bounds_classify+NUM_SCALE_LEVELS);
        *t7 = new thread(&cv::cuda::HOG::thread_fine_collect_locations, gpu_hog,
                &collect_locations_node, fine_init_barrier, t_info);
        //t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, cost_display);
        //t_info.phase;
        *t8 = new thread(&App::thread_display, this, &display_node,
                fine_init_barrier, g_idx == 0 && args.display);
    }

    /* graph construction finishes */

    printf("Joining pthreads...\n");

    for (int g_idx = 0; g_idx < args.num_fine_graphs; g_idx++) {
        graph_t g = arr_g[g_idx];
        thread* t0 = arr_t0[g_idx];
        thread* t1 = arr_t1[g_idx];
        thread** t2 = arr_t2 + g_idx * NUM_SCALE_LEVELS;
        thread** t34 = arr_t34 + g_idx * NUM_SCALE_LEVELS;
        thread** t5  = arr_t5 + g_idx * NUM_SCALE_LEVELS;
        thread** t6  = arr_t6 + g_idx * NUM_SCALE_LEVELS;
        thread* t7 = arr_t7[g_idx];
        thread* t8 = arr_t8[g_idx];
        t0->join();
        t1->join();
        delete t0;
        delete t1;
        for (int i=0; i<NUM_SCALE_LEVELS; i++) {
            if (t2[i]->joinable()) t2[i]->join();
            if (t34[i]->joinable()) t34[i]->join();
            if (t5[i]->joinable()) t5[i]->join();
            if (t6[i]->joinable()) t6[i]->join();

            delete t2[i];
            delete t34[i];
            delete t5[i];
            delete t6[i];
        }
        t7->join();
        t8->join();
        delete t7;
        delete t8;
        CheckError(pgm_destroy_graph(g));
    }
    free(arr_t1);
    free(arr_t2);
    free(arr_t34);
    free(arr_t5);
    free(arr_t6);
    free(arr_t7);
    free(arr_t8);

    //CheckError(pgm_destroy_graph(g));
    CheckError(pgm_destroy());
    fprintf(stdout, "cleaned up ...");
}

void App::sched_CD_hog(cv::Ptr<cv::cuda::HOG> gpu_hog, cv::HOGDescriptor cpu_hog, Mat* frames)
{
    pthread_barrier_t arr_fine_init_barrier[args.num_fine_graphs];
    /* graph construction */
    graph_t arr_g [args.num_fine_graphs];

    struct sync_info arr_sync_info_resize            [args.num_fine_graphs][NUM_SCALE_LEVELS];
    struct sync_info arr_sync_info_compute_gradients [args.num_fine_graphs][NUM_SCALE_LEVELS];
    struct sync_info arr_sync_info_CD                [args.num_fine_graphs][NUM_SCALE_LEVELS];
    struct sync_info arr_sync_info_classify          [args.num_fine_graphs][NUM_SCALE_LEVELS];

    thread** arr_t0  = (thread**) calloc(args.num_fine_graphs, sizeof(std::thread *));
    thread** arr_t1  = (thread**) calloc(args.num_fine_graphs, sizeof(std::thread *));
    thread** arr_t2  = (thread**) calloc(args.num_fine_graphs * NUM_SCALE_LEVELS, sizeof(std::thread *));
    thread** arr_t3  = (thread**) calloc(args.num_fine_graphs * NUM_SCALE_LEVELS, sizeof(std::thread *));
    thread** arr_t45 = (thread**) calloc(args.num_fine_graphs * NUM_SCALE_LEVELS, sizeof(std::thread *));
    thread** arr_t6  = (thread**) calloc(args.num_fine_graphs * NUM_SCALE_LEVELS, sizeof(std::thread *));
    thread** arr_t7  = (thread**) calloc(args.num_fine_graphs, sizeof(std::thread *));
    thread** arr_t8  = (thread**) calloc(args.num_fine_graphs, sizeof(std::thread *));

    char buf[30];
    sprintf(buf, "/tmp/graph_t%d", args.task_id);
    CheckError(pgm_init(buf, 1));

    for (int g_idx = 0; g_idx < args.num_fine_graphs; g_idx++)
    {
        pthread_barrier_t* fine_init_barrier = arr_fine_init_barrier + g_idx;

        // A graph consists of the graph itself, nodes, and edge
        graph_t* g_ptr = arr_g + g_idx;

        node_t color_convert_node;
        node_t compute_scales_node;
        node_t resize_node            [NUM_SCALE_LEVELS];
        node_t compute_gradients_node [NUM_SCALE_LEVELS];
        node_t CD_node                [NUM_SCALE_LEVELS]; // compute-hists + normalize-hists
        node_t classify_node          [NUM_SCALE_LEVELS];
        node_t collect_locations_node;
        node_t display_node;

        edge_t e0_1;
        edge_t e1_2  [NUM_SCALE_LEVELS];
        edge_t e2_3  [NUM_SCALE_LEVELS];
        edge_t e3_45 [NUM_SCALE_LEVELS];
        edge_t e45_6 [NUM_SCALE_LEVELS];
        edge_t e6_7  [NUM_SCALE_LEVELS];
        edge_t e7_8;

        // Initialize the graph
        sprintf(buf, "hog_%d", g_idx);
        CheckError(pgm_init_graph(g_ptr, buf));
        graph_t g = *g_ptr;

        // Initialize the nodes
        CheckError(pgm_init_node(&color_convert_node, g, "color_convert"));
        CheckError(pgm_init_node(&compute_scales_node, g, "compute_scales"));
        for (int i=0; i<NUM_SCALE_LEVELS; i++) {
            CheckError(pgm_init_node(resize_node + i, g, "resize"));
            CheckError(pgm_init_node(compute_gradients_node + i, g, "compute_gradients"));
            CheckError(pgm_init_node(CD_node + i, g, "CD"));
            CheckError(pgm_init_node(classify_node + i, g, "classify"));
        }
        CheckError(pgm_init_node(&collect_locations_node, g, "collect_locations"));
        CheckError(pgm_init_node(&display_node, g, "display"));

        // Initialize the edges
        edge_attr_t fast_mq_attr;
        memset(&fast_mq_attr, 0, sizeof(fast_mq_attr));
        fast_mq_attr.type = pgm_fast_fifo_edge;

        fast_mq_attr.nr_produce = sizeof(struct params_compute);
        fast_mq_attr.nr_consume = sizeof(struct params_compute);
        fast_mq_attr.nr_threshold = sizeof(struct params_compute);
        CheckError(pgm_init_edge(&e0_1, color_convert_node, compute_scales_node, "e0_1", &fast_mq_attr));

        for (int i=0; i<NUM_SCALE_LEVELS; i++) {
            sprintf(buf, "e1_2_%d", i);
            fast_mq_attr.nr_produce = sizeof(struct params_resize);
            fast_mq_attr.nr_consume = sizeof(struct params_resize);
            fast_mq_attr.nr_threshold = sizeof(struct params_resize);
            CheckError(pgm_init_edge(e1_2 + i,
                        compute_scales_node,
                        resize_node[i], buf, &fast_mq_attr));

            sprintf(buf, "e2_3_%d", i);
            fast_mq_attr.nr_produce = sizeof(struct params_compute_gradients);
            fast_mq_attr.nr_consume = sizeof(struct params_compute_gradients);
            fast_mq_attr.nr_threshold = sizeof(struct params_compute_gradients);
            CheckError(pgm_init_edge(e2_3 + i,
                        resize_node[i],
                        compute_gradients_node[i], buf, &fast_mq_attr));

            sprintf(buf, "e3_45_%d", i);
            fast_mq_attr.nr_produce = sizeof(struct params_compute_histograms);
            fast_mq_attr.nr_consume = sizeof(struct params_compute_histograms);
            fast_mq_attr.nr_threshold = sizeof(struct params_compute_histograms);
            CheckError(pgm_init_edge(e3_45 + i,
                        compute_gradients_node[i],
                        CD_node[i], buf, &fast_mq_attr));

            sprintf(buf, "e45_6_%d", i);
            fast_mq_attr.nr_produce = sizeof(struct params_fine_classify);
            fast_mq_attr.nr_consume = sizeof(struct params_fine_classify);
            fast_mq_attr.nr_threshold = sizeof(struct params_fine_classify);
            CheckError(pgm_init_edge(e45_6 + i,
                        CD_node[i],
                        classify_node[i], buf, &fast_mq_attr));

            sprintf(buf, "e6_7_%d", i);
            fast_mq_attr.nr_produce = sizeof(struct params_fine_collect_locations);
            fast_mq_attr.nr_consume = sizeof(struct params_fine_collect_locations);
            fast_mq_attr.nr_threshold = sizeof(struct params_fine_collect_locations);
            CheckError(pgm_init_edge(e6_7 + i,
                        classify_node[i],
                        collect_locations_node, buf, &fast_mq_attr));
        }

        fast_mq_attr.nr_produce = sizeof(struct params_display);
        fast_mq_attr.nr_consume = sizeof(struct params_display);
        fast_mq_attr.nr_threshold = sizeof(struct params_display);
        CheckError(pgm_init_edge(&e7_8, collect_locations_node, display_node, "e7_8", &fast_mq_attr));

        pthread_barrier_init(fine_init_barrier, 0, 4 * NUM_SCALE_LEVELS + 4);

        thread** t0  = arr_t0 + g_idx;
        thread** t1  = arr_t1 + g_idx;
        thread** t2  = arr_t2 + g_idx * NUM_SCALE_LEVELS;
        thread** t3  = arr_t3 + g_idx * NUM_SCALE_LEVELS;
        thread** t45 = arr_t45 + g_idx * NUM_SCALE_LEVELS;
        thread** t6  = arr_t6 + g_idx * NUM_SCALE_LEVELS;
        thread** t7  = arr_t7 + g_idx;
        thread** t8  = arr_t8 + g_idx;

        struct sync_info* in_sync_info_resize             = arr_sync_info_resize            [((g_idx + args.num_fine_graphs - 1) % args.num_fine_graphs)];
        struct sync_info* in_sync_info_compute_gradients  = arr_sync_info_compute_gradients [((g_idx + args.num_fine_graphs - 1) % args.num_fine_graphs)];
        struct sync_info* in_sync_info_CD                 = arr_sync_info_CD                [((g_idx + args.num_fine_graphs - 1) % args.num_fine_graphs)];
        struct sync_info* in_sync_info_classify           = arr_sync_info_classify          [((g_idx + args.num_fine_graphs - 1) % args.num_fine_graphs)];

        struct sync_info* out_sync_info_resize             = arr_sync_info_resize            [g_idx];
        struct sync_info* out_sync_info_compute_gradients  = arr_sync_info_compute_gradients [g_idx];
        struct sync_info* out_sync_info_CD                 = arr_sync_info_CD                [g_idx];
        struct sync_info* out_sync_info_classify           = arr_sync_info_classify          [g_idx];

        for (int i=0; i<NUM_SCALE_LEVELS; i++) {
            sync_info_init(i + in_sync_info_resize);
            sync_info_init(i + in_sync_info_compute_gradients);
            sync_info_init(i + in_sync_info_CD);
            sync_info_init(i + in_sync_info_classify);

            sync_info_init(i + out_sync_info_resize);
            sync_info_init(i + out_sync_info_compute_gradients);
            sync_info_init(i + out_sync_info_CD);
            sync_info_init(i + out_sync_info_classify);
        }

        /* G-FL fine-grained HOG */
        float bound_color_convert                   = 31.0194960215;
        float bound_compute_scales                  = 30.5688293615;
        float bounds_resize      [NUM_SCALE_LEVELS] = {30.2419972231,30.3148010256,30.2936775569,30.3079650324,30.2963991483,30.3058228698,30.3005336032,30.2888197835,30.293491626,30.2840584057,30.3013304497,30.3016323015,30.3111116087};
        float bounds_compute_grad[NUM_SCALE_LEVELS] = {30.4902863291,30.44552396,30.4706995604,30.4291495557,30.4171804989,30.4135009326,30.3785928987,30.4059076714,30.3816881109,30.3627509567,30.3564101342,30.3519205285,30.3405616838};
        // float bounds_compute_hist[NUM_SCALE_LEVELS] = {30.5326861275,30.4951636278,30.5415047381,30.4958614402,30.5149987735,30.4750125581,30.5373623676,30.522424222,30.4851937235,30.4577396346,30.4263855713,30.4118414514,30.5202006157};
        float bounds_CD          [NUM_SCALE_LEVELS] = {30.5326861275,30.4951636278,30.5415047381,30.4958614402,30.5149987735,30.4750125581,30.5373623676,30.522424222,30.4851937235,30.4577396346,30.4263855713,30.4118414514,30.5202006157};
        // float bounds_normalize   [NUM_SCALE_LEVELS] = {30.2903383654,30.3080869341,30.2928467608,30.3064130287,30.3114078316,30.3120987837,30.2985990076,30.2803770804,30.3099685825,30.3123524377,30.2989110618,30.2943410679,30.3081075149};
        float bounds_classify    [NUM_SCALE_LEVELS] = {30.4792191337,30.4948973087,30.4540269983,30.429525815,30.5229441951,30.5070612675,30.497249677,30.4955912512,30.4267037822,30.4508744413,30.430038752,30.3991297266,30.3767439685};
        //float bound_collect_locations               30.983976896;

        /* G-FL fine-grained HOG */
        float cost_color_convert                   = 4.509346;
        float cost_compute_scales                  = 1.947347;
        float costs_resize      [NUM_SCALE_LEVELS] = {0.089336,0.503219,0.383134,0.464357,0.398606,0.452179,0.42211,0.355518,0.382077,0.32845,0.42664,0.428356,0.482245};
        float costs_compute_grad[NUM_SCALE_LEVELS] = {1.500837,1.246367,1.389488,1.15328,1.085237,1.064319,0.86587,1.021152,0.883466,0.77581,0.739763,0.71424,0.649666};
        // float costs_compute_hist[NUM_SCALE_LEVELS] = {1.741876,1.528564,1.792009,1.532531,1.641325,1.414007,1.76846,1.683538,1.471886,1.315812,1.137567,1.054885,1.670897};
        float costs_CD          [NUM_SCALE_LEVELS] = {1.741876,1.528564,1.792009,1.532531,1.641325,1.414007,1.76846,1.683538,1.471886,1.315812,1.137567,1.054885,1.670897};
        // float costs_normalize   [NUM_SCALE_LEVELS] = {0.364151,0.46505,0.378411,0.455534,0.483929,0.487857,0.411112,0.307522,0.475747,0.489299,0.412886,0.386906,0.465167};
        float costs_classify    [NUM_SCALE_LEVELS] = {1.437921,1.52705,1.294706,1.155419,1.686494,1.596201,1.540423,1.530995,1.139376,1.276784,1.158335,0.98262,0.855359};
        float cost_collect_locations               = 4.307423;

        /* C-FL fine-grained HOG */
        //float bound_color_convert                   = 26.3496352098;
        //float bound_compute_scales                  = 25.6841982494;
        //float bounds_resize      [NUM_SCALE_LEVELS] = {25.0939476981,25.1387935171,25.130189875,25.1357033275,25.1312621102,25.1463332249,25.1316714257,25.1302245094,25.1267072576,25.1262401226,25.1348909935,25.1432006726,25.1413123823};
        //float bounds_compute_grad[NUM_SCALE_LEVELS] = {25.3489965402,25.4185896374,25.4092134479,25.4079932294,25.4051932818,25.4030496702,25.312479578,25.291207475,25.2868561354,25.2685316772,25.2518075571,25.246440943,25.2364255915};
        //float bounds_compute_hist[NUM_SCALE_LEVELS] = {25.5728014354,25.4973631396,25.5123077387,25.471628638,25.3841716342,25.3274278981,25.4720253593,25.5328196023,25.4654399579,25.3426074958,25.4860697502,25.3209827517,25.4610373822};
        // float bounds_CD         [NUM_SCALE_LEVELS] = {25.5728014354,25.4973631396,25.5123077387,25.471628638,25.3841716342,25.3274278981,25.4720253593,25.5328196023,25.4654399579,25.3426074958,25.4860697502,25.3209827517,25.4610373822};
        //float bounds_normalize   [NUM_SCALE_LEVELS] = {25.1658996554,25.1786894798,25.1487118351,25.1680595823,25.1750299693,25.1885187788,25.1646279147,25.1649092834,25.1582577618,25.1765865135,25.1940356661,25.1748711091,25.179600851};
        //float bounds_classify    [NUM_SCALE_LEVELS] = {25.2909527262,25.3017792668,25.4537335318,25.3216511097,25.4693284561,25.3272781973,25.3334016159,25.3265829333,25.3061011243,25.2757711248,25.2811148401,25.2748319888,25.3276382806};
        ////float bound_collect_locations               26.204139258;

        /////* C-FL fine-grained HOG */
        //float cost_color_convert                   = 4.475663;
        //float cost_compute_scales                  = 2.150868;
        //float costs_resize      [NUM_SCALE_LEVELS] = {0.088747,0.245422,0.215364,0.234626,0.21911,0.271763,0.22054,0.215485,0.203197,0.201565,0.231788,0.260819,0.254222};
        //float costs_compute_grad[NUM_SCALE_LEVELS] = {0.979795,1.222928,1.190171,1.185908,1.176126,1.168637,0.852218,0.777901,0.762699,0.69868,0.640252,0.621503,0.586513};
        // float costs_compute_hist[NUM_SCALE_LEVELS] = {1.761688,1.498134,1.550345,1.408227,1.102684,0.904442,1.409613,1.622006,1.386606,0.957474,1.458679,0.881925,1.371225};
        //float costs_CD          [NUM_SCALE_LEVELS] = {1.761688,1.498134,1.550345,1.408227,1.102684,0.904442,1.409613,1.622006,1.386606,0.957474,1.458679,0.881925,1.371225};
        //float costs_normalize   [NUM_SCALE_LEVELS] = {0.340121,0.384804,0.280073,0.347667,0.372019,0.419144,0.335678,0.336661,0.313423,0.377457,0.438418,0.371464,0.387988};
        //float costs_classify    [NUM_SCALE_LEVELS] = {0.777011,0.814835,1.345708,0.88426,1.400191,0.903919,0.925312,0.90149,0.829934,0.723972,0.742641,0.720691,0.905177};
        //float cost_collect_locations               = 3.967353;

        /* | first graph release      | second graph release     | first graph release again
         *  <---------PERIOD--------->
         *  <--------------- PERIOD * args.num_fine_graphs ---------->
         */
        int period = PERIOD * args.num_fine_graphs;
        int m_cpus = 16;
        struct task_info t_info;
        t_info.early = args.early;
        t_info.realtime = args.realtime;
        t_info.sched = fine_CD;
        t_info.period = period;
        t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, cost_color_convert);
        t_info.phase = PERIOD * g_idx;
        t_info.id = 0;
        if (args.cluster != -1)
            t_info.cluster = args.cluster;
        else
            t_info.cluster = args.cluster;
        *t0 = new thread(&App::thread_color_convert, this,
                &color_convert_node, fine_init_barrier,
                gpu_hog, cpu_hog, frames, t_info, g_idx);

        t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, cost_compute_scales);
        t_info.id = 1;
        t_info.phase = t_info.phase + bound_color_convert;
        *t1 = new thread(&cv::cuda::HOG::thread_fine_compute_scales, gpu_hog,
                &compute_scales_node, fine_init_barrier, t_info);

        for (int i=0; i<NUM_SCALE_LEVELS; i++) {
            t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, costs_resize[i]);
            t_info.id = 0 * NUM_SCALE_LEVELS + i + 2;
            t_info.phase = PERIOD * g_idx + bound_color_convert + bound_compute_scales;
            t_info.s_info_in = in_sync_info_resize + i;
            t_info.s_info_out = out_sync_info_resize + i;
            t2[i] = new thread(&cv::cuda::HOG::thread_fine_resize, gpu_hog,
                    resize_node + i, fine_init_barrier, t_info);

            t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, costs_compute_grad[i]);
            t_info.id = 1 * NUM_SCALE_LEVELS + i + 2;
            t_info.phase = t_info.phase + bounds_resize[i];
            t_info.s_info_in = in_sync_info_compute_gradients + i;
            t_info.s_info_out = out_sync_info_compute_gradients + i;
            t3[i] = new thread(&cv::cuda::HOG::thread_fine_compute_gradients,
                    gpu_hog, compute_gradients_node + i, fine_init_barrier, t_info);

            t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, costs_CD[i]);
            t_info.id = 2 * NUM_SCALE_LEVELS + i + 2;
            t_info.phase = t_info.phase + bounds_compute_grad[i];
            t_info.s_info_in = in_sync_info_CD + i;
            t_info.s_info_out = out_sync_info_CD + i;
            t45[i] = new thread(&cv::cuda::HOG::thread_fine_CD,
                    gpu_hog, CD_node + i, fine_init_barrier, t_info);

            t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, costs_classify[i]);
            t_info.id = 3 * NUM_SCALE_LEVELS + i + 2;
            t_info.phase = t_info.phase + bounds_CD[i];
            t_info.s_info_in = in_sync_info_classify + i;
            t_info.s_info_out = out_sync_info_classify + i;
            t6[i] = new thread(&cv::cuda::HOG::thread_fine_classify, gpu_hog,
                    classify_node + i, fine_init_barrier, t_info);
        }

        t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, cost_collect_locations);
        t_info.id = 4 * NUM_SCALE_LEVELS + 2;
        t_info.phase = t_info.phase + *std::max_element(bounds_classify, bounds_classify+NUM_SCALE_LEVELS);
        *t7 = new thread(&cv::cuda::HOG::thread_fine_collect_locations, gpu_hog,
                &collect_locations_node, fine_init_barrier, t_info);
        //t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, cost_display);
        //t_info.phase;
        *t8 = new thread(&App::thread_display, this, &display_node,
                fine_init_barrier, g_idx == 0 && args.display);
    }

    /* graph construction finishes */

    printf("Joining pthreads...\n");

    for (int g_idx = 0; g_idx < args.num_fine_graphs; g_idx++) {
        graph_t g = arr_g[g_idx];
        thread* t0 = arr_t0[g_idx];
        thread* t1 = arr_t1[g_idx];
        thread** t2 = arr_t2 + g_idx * NUM_SCALE_LEVELS;
        thread** t3 = arr_t3 + g_idx * NUM_SCALE_LEVELS;
        thread** t45 = arr_t45 + g_idx * NUM_SCALE_LEVELS;
        thread** t6  = arr_t6 + g_idx * NUM_SCALE_LEVELS;
        thread* t7 = arr_t7[g_idx];
        thread* t8 = arr_t8[g_idx];
        t0->join();
        t1->join();
        delete t0;
        delete t1;
        for (int i=0; i<NUM_SCALE_LEVELS; i++) {
            if (t2[i]->joinable()) t2[i]->join();
            if (t3[i]->joinable()) t3[i]->join();
            if (t45[i]->joinable()) t45[i]->join();
            if (t6[i]->joinable()) t6[i]->join();

            delete t2[i];
            delete t3[i];
            delete t45[i];
            delete t6[i];
        }
        t7->join();
        t8->join();
        delete t7;
        delete t8;
        CheckError(pgm_destroy_graph(g));
    }
    free(arr_t1);
    free(arr_t2);
    free(arr_t3);
    free(arr_t45);
    free(arr_t6);
    free(arr_t7);
    free(arr_t8);

    //CheckError(pgm_destroy_graph(g));
    CheckError(pgm_destroy());
    fprintf(stdout, "cleaned up ...");
}

void App::sched_DE_hog(cv::Ptr<cv::cuda::HOG> gpu_hog, cv::HOGDescriptor cpu_hog, Mat* frames)
{
    pthread_barrier_t arr_fine_init_barrier[args.num_fine_graphs];
    /* graph construction */
    graph_t arr_g [args.num_fine_graphs];

    struct sync_info arr_sync_info_resize             [args.num_fine_graphs][NUM_SCALE_LEVELS];
    struct sync_info arr_sync_info_compute_gradients  [args.num_fine_graphs][NUM_SCALE_LEVELS];
    struct sync_info arr_sync_info_compute_histograms [args.num_fine_graphs][NUM_SCALE_LEVELS];
    struct sync_info arr_sync_info_DE                 [args.num_fine_graphs][NUM_SCALE_LEVELS];

    thread** arr_t0  = (thread**) calloc(args.num_fine_graphs, sizeof(std::thread *));
    thread** arr_t1  = (thread**) calloc(args.num_fine_graphs, sizeof(std::thread *));
    thread** arr_t2  = (thread**) calloc(args.num_fine_graphs * NUM_SCALE_LEVELS, sizeof(std::thread *));
    thread** arr_t3  = (thread**) calloc(args.num_fine_graphs * NUM_SCALE_LEVELS, sizeof(std::thread *));
    thread** arr_t4  = (thread**) calloc(args.num_fine_graphs * NUM_SCALE_LEVELS, sizeof(std::thread *));
    thread** arr_t56 = (thread**) calloc(args.num_fine_graphs * NUM_SCALE_LEVELS, sizeof(std::thread *));
    thread** arr_t7  = (thread**) calloc(args.num_fine_graphs, sizeof(std::thread *));
    thread** arr_t8  = (thread**) calloc(args.num_fine_graphs, sizeof(std::thread *));

    char buf[30];
    sprintf(buf, "/tmp/graph_t%d", args.task_id);
    CheckError(pgm_init(buf, 1));

    for (int g_idx = 0; g_idx < args.num_fine_graphs; g_idx++)
    {
        pthread_barrier_t* fine_init_barrier = arr_fine_init_barrier + g_idx;

        // A graph consists of the graph itself, nodes, and edge
        graph_t* g_ptr = arr_g + g_idx;

        node_t color_convert_node;
        node_t compute_scales_node;
        node_t resize_node             [NUM_SCALE_LEVELS];
        node_t compute_gradients_node  [NUM_SCALE_LEVELS];
        node_t compute_histograms_node [NUM_SCALE_LEVELS];
        node_t DE_node                 [NUM_SCALE_LEVELS]; // normalize-hists + classify-hists
        node_t collect_locations_node;
        node_t display_node;

        edge_t e0_1;
        edge_t e1_2  [NUM_SCALE_LEVELS];
        edge_t e2_3  [NUM_SCALE_LEVELS];
        edge_t e3_4 [NUM_SCALE_LEVELS];
        edge_t e4_56 [NUM_SCALE_LEVELS];
        edge_t e56_7  [NUM_SCALE_LEVELS];
        edge_t e7_8;

        // Initialize the graph
        sprintf(buf, "hog_%d", g_idx);
        CheckError(pgm_init_graph(g_ptr, buf));
        graph_t g = *g_ptr;

        // Initialize the nodes
        CheckError(pgm_init_node(&color_convert_node, g, "color_convert"));
        CheckError(pgm_init_node(&compute_scales_node, g, "compute_scales"));
        for (int i=0; i<NUM_SCALE_LEVELS; i++) {
            CheckError(pgm_init_node(resize_node + i, g, "resize"));
            CheckError(pgm_init_node(compute_gradients_node + i, g, "compute_gradients"));
            CheckError(pgm_init_node(compute_histograms_node + i, g, "compute_histograms"));
            CheckError(pgm_init_node(DE_node + i, g, "DE"));
        }
        CheckError(pgm_init_node(&collect_locations_node, g, "collect_locations"));
        CheckError(pgm_init_node(&display_node, g, "display"));

        // Initialize the edges
        edge_attr_t fast_mq_attr;
        memset(&fast_mq_attr, 0, sizeof(fast_mq_attr));
        fast_mq_attr.type = pgm_fast_fifo_edge;

        fast_mq_attr.nr_produce = sizeof(struct params_compute);
        fast_mq_attr.nr_consume = sizeof(struct params_compute);
        fast_mq_attr.nr_threshold = sizeof(struct params_compute);
        CheckError(pgm_init_edge(&e0_1, color_convert_node, compute_scales_node, "e0_1", &fast_mq_attr));

        for (int i=0; i<NUM_SCALE_LEVELS; i++) {
            sprintf(buf, "e1_2_%d", i);
            fast_mq_attr.nr_produce = sizeof(struct params_resize);
            fast_mq_attr.nr_consume = sizeof(struct params_resize);
            fast_mq_attr.nr_threshold = sizeof(struct params_resize);
            CheckError(pgm_init_edge(e1_2 + i,
                        compute_scales_node,
                        resize_node[i], buf, &fast_mq_attr));

            sprintf(buf, "e2_3_%d", i);
            fast_mq_attr.nr_produce = sizeof(struct params_compute_gradients);
            fast_mq_attr.nr_consume = sizeof(struct params_compute_gradients);
            fast_mq_attr.nr_threshold = sizeof(struct params_compute_gradients);
            CheckError(pgm_init_edge(e2_3 + i,
                        resize_node[i],
                        compute_gradients_node[i], buf, &fast_mq_attr));

            sprintf(buf, "e3_4_%d", i);
            fast_mq_attr.nr_produce = sizeof(struct params_compute_histograms);
            fast_mq_attr.nr_consume = sizeof(struct params_compute_histograms);
            fast_mq_attr.nr_threshold = sizeof(struct params_compute_histograms);
            CheckError(pgm_init_edge(e3_4 + i,
                        compute_gradients_node[i],
                        compute_histograms_node[i], buf, &fast_mq_attr));

            sprintf(buf, "e4_56_%d", i);
            fast_mq_attr.nr_produce = sizeof(struct params_fine_normalize);
            fast_mq_attr.nr_consume = sizeof(struct params_fine_normalize);
            fast_mq_attr.nr_threshold = sizeof(struct params_fine_normalize);
            CheckError(pgm_init_edge(e4_56 + i,
                        compute_histograms_node[i],
                        DE_node[i], buf, &fast_mq_attr));

            sprintf(buf, "e56_7_%d", i);
            fast_mq_attr.nr_produce = sizeof(struct params_fine_collect_locations);
            fast_mq_attr.nr_consume = sizeof(struct params_fine_collect_locations);
            fast_mq_attr.nr_threshold = sizeof(struct params_fine_collect_locations);
            CheckError(pgm_init_edge(e56_7 + i,
                        DE_node[i],
                        collect_locations_node, buf, &fast_mq_attr));
        }

        fast_mq_attr.nr_produce = sizeof(struct params_display);
        fast_mq_attr.nr_consume = sizeof(struct params_display);
        fast_mq_attr.nr_threshold = sizeof(struct params_display);
        CheckError(pgm_init_edge(&e7_8, collect_locations_node, display_node, "e7_8", &fast_mq_attr));

        pthread_barrier_init(fine_init_barrier, 0, 4 * NUM_SCALE_LEVELS + 4);

        thread** t0  = arr_t0 + g_idx;
        thread** t1  = arr_t1 + g_idx;
        thread** t2  = arr_t2 + g_idx * NUM_SCALE_LEVELS;
        thread** t3  = arr_t3 + g_idx * NUM_SCALE_LEVELS;
        thread** t4  = arr_t4 + g_idx * NUM_SCALE_LEVELS;
        thread** t56 = arr_t56 + g_idx * NUM_SCALE_LEVELS;
        thread** t7  = arr_t7 + g_idx;
        thread** t8  = arr_t8 + g_idx;

        struct sync_info* in_sync_info_resize             = arr_sync_info_resize            [((g_idx + args.num_fine_graphs - 1) % args.num_fine_graphs)];
        struct sync_info* in_sync_info_compute_gradients  = arr_sync_info_compute_gradients [((g_idx + args.num_fine_graphs - 1) % args.num_fine_graphs)];
        struct sync_info* in_sync_info_compute_histograms = arr_sync_info_compute_histograms[((g_idx + args.num_fine_graphs - 1) % args.num_fine_graphs)];
        struct sync_info* in_sync_info_DE                 = arr_sync_info_DE                [((g_idx + args.num_fine_graphs - 1) % args.num_fine_graphs)];

        struct sync_info* out_sync_info_resize             = arr_sync_info_resize            [g_idx];
        struct sync_info* out_sync_info_compute_gradients  = arr_sync_info_compute_gradients [g_idx];
        struct sync_info* out_sync_info_compute_histograms = arr_sync_info_compute_histograms[g_idx];
        struct sync_info* out_sync_info_DE                 = arr_sync_info_DE                [g_idx];

        for (int i=0; i<NUM_SCALE_LEVELS; i++) {
            sync_info_init(i + in_sync_info_resize);
            sync_info_init(i + in_sync_info_compute_gradients);
            sync_info_init(i + in_sync_info_compute_histograms);
            sync_info_init(i + in_sync_info_DE);

            sync_info_init(i + out_sync_info_resize);
            sync_info_init(i + out_sync_info_compute_gradients);
            sync_info_init(i + out_sync_info_compute_histograms);
            sync_info_init(i + out_sync_info_DE);
        }

        /* G-FL fine-grained HOG */
        float bound_color_convert                   = 31.0194960215;
        float bound_compute_scales                  = 30.5688293615;
        float bounds_resize      [NUM_SCALE_LEVELS] = {30.2419972231,30.3148010256,30.2936775569,30.3079650324,30.2963991483,30.3058228698,30.3005336032,30.2888197835,30.293491626,30.2840584057,30.3013304497,30.3016323015,30.3111116087};
        float bounds_compute_grad[NUM_SCALE_LEVELS] = {30.4902863291,30.44552396,30.4706995604,30.4291495557,30.4171804989,30.4135009326,30.3785928987,30.4059076714,30.3816881109,30.3627509567,30.3564101342,30.3519205285,30.3405616838};
        float bounds_compute_hist[NUM_SCALE_LEVELS] = {30.5326861275,30.4951636278,30.5415047381,30.4958614402,30.5149987735,30.4750125581,30.5373623676,30.522424222,30.4851937235,30.4577396346,30.4263855713,30.4118414514,30.5202006157};
        // float bounds_normalize   [NUM_SCALE_LEVELS] = {30.2903383654,30.3080869341,30.2928467608,30.3064130287,30.3114078316,30.3120987837,30.2985990076,30.2803770804,30.3099685825,30.3123524377,30.2989110618,30.2943410679,30.3081075149};
        float bounds_DE          [NUM_SCALE_LEVELS] = {30.2903383654,30.3080869341,30.2928467608,30.3064130287,30.3114078316,30.3120987837,30.2985990076,30.2803770804,30.3099685825,30.3123524377,30.2989110618,30.2943410679,30.3081075149};
        // float bounds_classify    [NUM_SCALE_LEVELS] = {30.4792191337,30.4948973087,30.4540269983,30.429525815,30.5229441951,30.5070612675,30.497249677,30.4955912512,30.4267037822,30.4508744413,30.430038752,30.3991297266,30.3767439685};
        //float bound_collect_locations               30.983976896;

        /* G-FL fine-grained HOG */
        float cost_color_convert                   = 4.509346;
        float cost_compute_scales                  = 1.947347;
        float costs_resize      [NUM_SCALE_LEVELS] = {0.089336,0.503219,0.383134,0.464357,0.398606,0.452179,0.42211,0.355518,0.382077,0.32845,0.42664,0.428356,0.482245};
        float costs_compute_grad[NUM_SCALE_LEVELS] = {1.500837,1.246367,1.389488,1.15328,1.085237,1.064319,0.86587,1.021152,0.883466,0.77581,0.739763,0.71424,0.649666};
        float costs_compute_hist[NUM_SCALE_LEVELS] = {1.741876,1.528564,1.792009,1.532531,1.641325,1.414007,1.76846,1.683538,1.471886,1.315812,1.137567,1.054885,1.670897};
        // float costs_normalize   [NUM_SCALE_LEVELS] = {0.364151,0.46505,0.378411,0.455534,0.483929,0.487857,0.411112,0.307522,0.475747,0.489299,0.412886,0.386906,0.465167};
        float costs_DE          [NUM_SCALE_LEVELS] = {0.364151,0.46505,0.378411,0.455534,0.483929,0.487857,0.411112,0.307522,0.475747,0.489299,0.412886,0.386906,0.465167};
        // float costs_classify    [NUM_SCALE_LEVELS] = {1.437921,1.52705,1.294706,1.155419,1.686494,1.596201,1.540423,1.530995,1.139376,1.276784,1.158335,0.98262,0.855359};
        float cost_collect_locations               = 4.307423;

        /* C-FL fine-grained HOG */
        //float bound_color_convert                   = 26.3496352098;
        //float bound_compute_scales                  = 25.6841982494;
        //float bounds_resize      [NUM_SCALE_LEVELS] = {25.0939476981,25.1387935171,25.130189875,25.1357033275,25.1312621102,25.1463332249,25.1316714257,25.1302245094,25.1267072576,25.1262401226,25.1348909935,25.1432006726,25.1413123823};
        //float bounds_compute_grad[NUM_SCALE_LEVELS] = {25.3489965402,25.4185896374,25.4092134479,25.4079932294,25.4051932818,25.4030496702,25.312479578,25.291207475,25.2868561354,25.2685316772,25.2518075571,25.246440943,25.2364255915};
        //float bounds_compute_hist[NUM_SCALE_LEVELS] = {25.5728014354,25.4973631396,25.5123077387,25.471628638,25.3841716342,25.3274278981,25.4720253593,25.5328196023,25.4654399579,25.3426074958,25.4860697502,25.3209827517,25.4610373822};
        //float bounds_normalize   [NUM_SCALE_LEVELS] = {25.1658996554,25.1786894798,25.1487118351,25.1680595823,25.1750299693,25.1885187788,25.1646279147,25.1649092834,25.1582577618,25.1765865135,25.1940356661,25.1748711091,25.179600851};
        //float bounds_DE          [NUM_SCALE_LEVELS] = {25.1658996554,25.1786894798,25.1487118351,25.1680595823,25.1750299693,25.1885187788,25.1646279147,25.1649092834,25.1582577618,25.1765865135,25.1940356661,25.1748711091,25.179600851};
        //float bounds_classify    [NUM_SCALE_LEVELS] = {25.2909527262,25.3017792668,25.4537335318,25.3216511097,25.4693284561,25.3272781973,25.3334016159,25.3265829333,25.3061011243,25.2757711248,25.2811148401,25.2748319888,25.3276382806};
        ////float bound_collect_locations               26.204139258;

        /////* C-FL fine-grained HOG */
        //float cost_color_convert                   = 4.475663;
        //float cost_compute_scales                  = 2.150868;
        //float costs_resize      [NUM_SCALE_LEVELS] = {0.088747,0.245422,0.215364,0.234626,0.21911,0.271763,0.22054,0.215485,0.203197,0.201565,0.231788,0.260819,0.254222};
        //float costs_compute_grad[NUM_SCALE_LEVELS] = {0.979795,1.222928,1.190171,1.185908,1.176126,1.168637,0.852218,0.777901,0.762699,0.69868,0.640252,0.621503,0.586513};
        //float costs_compute_hist[NUM_SCALE_LEVELS] = {1.761688,1.498134,1.550345,1.408227,1.102684,0.904442,1.409613,1.622006,1.386606,0.957474,1.458679,0.881925,1.371225};
        //float costs_normalize   [NUM_SCALE_LEVELS] = {0.340121,0.384804,0.280073,0.347667,0.372019,0.419144,0.335678,0.336661,0.313423,0.377457,0.438418,0.371464,0.387988};
        //float costs_DE          [NUM_SCALE_LEVELS] = {0.340121,0.384804,0.280073,0.347667,0.372019,0.419144,0.335678,0.336661,0.313423,0.377457,0.438418,0.371464,0.387988};
        //float costs_classify    [NUM_SCALE_LEVELS] = {0.777011,0.814835,1.345708,0.88426,1.400191,0.903919,0.925312,0.90149,0.829934,0.723972,0.742641,0.720691,0.905177};
        //float cost_collect_locations               = 3.967353;

        /* | first graph release      | second graph release     | first graph release again
         *  <---------PERIOD--------->
         *  <--------------- PERIOD * args.num_fine_graphs ---------->
         */
        int period = PERIOD * args.num_fine_graphs;
        int m_cpus = 16;
        struct task_info t_info;
        t_info.early = args.early;
        t_info.realtime = args.realtime;
        t_info.sched = fine_DE;
        t_info.period = period;
        t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, cost_color_convert);
        t_info.phase = PERIOD * g_idx;
        t_info.id = 0;
        if (args.cluster != -1)
            t_info.cluster = args.cluster;
        else
            t_info.cluster = args.cluster;
        *t0 = new thread(&App::thread_color_convert, this,
                &color_convert_node, fine_init_barrier,
                gpu_hog, cpu_hog, frames, t_info, g_idx);

        t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, cost_compute_scales);
        t_info.id = 1;
        t_info.phase = t_info.phase + bound_color_convert;
        *t1 = new thread(&cv::cuda::HOG::thread_fine_compute_scales, gpu_hog,
                &compute_scales_node, fine_init_barrier, t_info);

        for (int i=0; i<NUM_SCALE_LEVELS; i++) {
            t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, costs_resize[i]);
            t_info.id = 0 * NUM_SCALE_LEVELS + i + 2;
            t_info.phase = PERIOD * g_idx + bound_color_convert + bound_compute_scales;
            t_info.s_info_in = in_sync_info_resize + i;
            t_info.s_info_out = out_sync_info_resize + i;
            t2[i] = new thread(&cv::cuda::HOG::thread_fine_resize, gpu_hog,
                    resize_node + i, fine_init_barrier, t_info);

            t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, costs_compute_grad[i]);
            t_info.id = 1 * NUM_SCALE_LEVELS + i + 2;
            t_info.phase = t_info.phase + bounds_resize[i];
            t_info.s_info_in = in_sync_info_compute_gradients + i;
            t_info.s_info_out = out_sync_info_compute_gradients + i;
            t3[i] = new thread(&cv::cuda::HOG::thread_fine_compute_gradients,
                    gpu_hog, compute_gradients_node + i, fine_init_barrier, t_info);

            t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, costs_compute_hist[i]);
            t_info.id = 2 * NUM_SCALE_LEVELS + i + 2;
            t_info.phase = t_info.phase + bounds_compute_grad[i];
            t_info.s_info_in = in_sync_info_compute_histograms + i;
            t_info.s_info_out = out_sync_info_compute_histograms + i;
            t4[i] = new thread(&cv::cuda::HOG::thread_fine_compute_histograms,
                    gpu_hog, compute_histograms_node + i, fine_init_barrier,
                    t_info);

            t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, costs_DE[i]);
            t_info.id = 3 * NUM_SCALE_LEVELS + i + 2;
            t_info.phase = t_info.phase + bounds_compute_hist[i];
            t_info.s_info_in = in_sync_info_DE + i;
            t_info.s_info_out = out_sync_info_DE + i;
            t56[i] = new thread(&cv::cuda::HOG::thread_fine_DE, gpu_hog,
                    DE_node + i, fine_init_barrier, t_info);
        }

        t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, cost_collect_locations);
        t_info.id = 4 * NUM_SCALE_LEVELS + 2;
        t_info.phase = t_info.phase + *std::max_element(bounds_DE, bounds_DE+NUM_SCALE_LEVELS);
        *t7 = new thread(&cv::cuda::HOG::thread_fine_collect_locations, gpu_hog,
                &collect_locations_node, fine_init_barrier, t_info);
        //t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, cost_display);
        //t_info.phase;
        *t8 = new thread(&App::thread_display, this, &display_node,
                fine_init_barrier, g_idx == 0 && args.display);
    }

    /* graph construction finishes */

    printf("Joining pthreads...\n");

    for (int g_idx = 0; g_idx < args.num_fine_graphs; g_idx++) {
        graph_t g = arr_g[g_idx];
        thread* t0 = arr_t0[g_idx];
        thread* t1 = arr_t1[g_idx];
        thread** t2 = arr_t2 + g_idx * NUM_SCALE_LEVELS;
        thread** t3 = arr_t3 + g_idx * NUM_SCALE_LEVELS;
        thread** t4 = arr_t4 + g_idx * NUM_SCALE_LEVELS;
        thread** t56  = arr_t56 + g_idx * NUM_SCALE_LEVELS;
        thread* t7 = arr_t7[g_idx];
        thread* t8 = arr_t8[g_idx];
        t0->join();
        t1->join();
        delete t0;
        delete t1;
        for (int i=0; i<NUM_SCALE_LEVELS; i++) {
            if (t2[i]->joinable()) t2[i]->join();
            if (t3[i]->joinable()) t3[i]->join();
            if (t4[i]->joinable()) t4[i]->join();
            if (t56[i]->joinable()) t56[i]->join();

            delete t2[i];
            delete t3[i];
            delete t4[i];
            delete t56[i];
        }
        t7->join();
        t8->join();
        delete t7;
        delete t8;
        CheckError(pgm_destroy_graph(g));
    }
    free(arr_t1);
    free(arr_t2);
    free(arr_t3);
    free(arr_t4);
    free(arr_t56);
    free(arr_t7);
    free(arr_t8);

    //CheckError(pgm_destroy_graph(g));
    CheckError(pgm_destroy());
    fprintf(stdout, "cleaned up ...");
}

void App::sched_ABC_hog(cv::Ptr<cv::cuda::HOG> gpu_hog, cv::HOGDescriptor cpu_hog, Mat* frames)
{
    pthread_barrier_t arr_fine_init_barrier[args.num_fine_graphs];
    /* graph construction */
    graph_t arr_g [args.num_fine_graphs];

    struct sync_info arr_sync_info_ABC       [args.num_fine_graphs][NUM_SCALE_LEVELS];
    struct sync_info arr_sync_info_normalize [args.num_fine_graphs][NUM_SCALE_LEVELS];
    struct sync_info arr_sync_info_classify  [args.num_fine_graphs][NUM_SCALE_LEVELS];

    thread** arr_t0   = (thread**) calloc(args.num_fine_graphs, sizeof(std::thread *));
    thread** arr_t1   = (thread**) calloc(args.num_fine_graphs, sizeof(std::thread *));
    thread** arr_t234 = (thread**) calloc(args.num_fine_graphs * NUM_SCALE_LEVELS, sizeof(std::thread *));
    thread** arr_t5   = (thread**) calloc(args.num_fine_graphs * NUM_SCALE_LEVELS, sizeof(std::thread *));
    thread** arr_t6   = (thread**) calloc(args.num_fine_graphs * NUM_SCALE_LEVELS, sizeof(std::thread *));
    thread** arr_t7   = (thread**) calloc(args.num_fine_graphs, sizeof(std::thread *));
    thread** arr_t8   = (thread**) calloc(args.num_fine_graphs, sizeof(std::thread *));

    char buf[30];
    sprintf(buf, "/tmp/graph_t%d", args.task_id);
    CheckError(pgm_init(buf, 1));

    for (int g_idx = 0; g_idx < args.num_fine_graphs; g_idx++)
    {
        pthread_barrier_t* fine_init_barrier = arr_fine_init_barrier + g_idx;

        // A graph consists of the graph itself, nodes, and edge
        graph_t* g_ptr = arr_g + g_idx;

        node_t color_convert_node;
        node_t compute_scales_node;
        node_t ABC_node                [NUM_SCALE_LEVELS]; // resize -> compute-hists
        node_t normalize_node          [NUM_SCALE_LEVELS];
        node_t classify_node           [NUM_SCALE_LEVELS];
        node_t collect_locations_node;
        node_t display_node;

        edge_t e0_1;
        edge_t e1_234 [NUM_SCALE_LEVELS];
        edge_t e234_5 [NUM_SCALE_LEVELS];
        edge_t e5_6   [NUM_SCALE_LEVELS];
        edge_t e6_7   [NUM_SCALE_LEVELS];
        edge_t e7_8;

        // Initialize the graph
        sprintf(buf, "hog_%d", g_idx);
        CheckError(pgm_init_graph(g_ptr, buf));
        graph_t g = *g_ptr;

        // Initialize the nodes
        CheckError(pgm_init_node(&color_convert_node, g, "color_convert"));
        CheckError(pgm_init_node(&compute_scales_node, g, "compute_scales"));
        for (int i=0; i<NUM_SCALE_LEVELS; i++) {
            CheckError(pgm_init_node(ABC_node + i, g, "ABC"));
            CheckError(pgm_init_node(normalize_node + i, g, "normalize"));
            CheckError(pgm_init_node(classify_node + i, g, "classify"));
        }
        CheckError(pgm_init_node(&collect_locations_node, g, "collect_locations"));
        CheckError(pgm_init_node(&display_node, g, "display"));

        // Initialize the edges
        edge_attr_t fast_mq_attr;
        memset(&fast_mq_attr, 0, sizeof(fast_mq_attr));
        fast_mq_attr.type = pgm_fast_fifo_edge;

        fast_mq_attr.nr_produce = sizeof(struct params_compute);
        fast_mq_attr.nr_consume = sizeof(struct params_compute);
        fast_mq_attr.nr_threshold = sizeof(struct params_compute);
        CheckError(pgm_init_edge(&e0_1, color_convert_node, compute_scales_node, "e0_1", &fast_mq_attr));

        for (int i=0; i<NUM_SCALE_LEVELS; i++) {
            sprintf(buf, "e1_234_%d", i);
            fast_mq_attr.nr_produce = sizeof(struct params_resize);
            fast_mq_attr.nr_consume = sizeof(struct params_resize);
            fast_mq_attr.nr_threshold = sizeof(struct params_resize);
            CheckError(pgm_init_edge(e1_234 + i,
                        compute_scales_node,
                        ABC_node[i], buf, &fast_mq_attr));

            sprintf(buf, "e234_5_%d", i);
            fast_mq_attr.nr_produce = sizeof(struct params_fine_normalize);
            fast_mq_attr.nr_consume = sizeof(struct params_fine_normalize);
            fast_mq_attr.nr_threshold = sizeof(struct params_fine_normalize);
            CheckError(pgm_init_edge(e234_5 + i,
                        ABC_node[i],
                        normalize_node[i], buf, &fast_mq_attr));

            sprintf(buf, "e5_6_%d", i);
            fast_mq_attr.nr_produce = sizeof(struct params_fine_classify);
            fast_mq_attr.nr_consume = sizeof(struct params_fine_classify);
            fast_mq_attr.nr_threshold = sizeof(struct params_fine_classify);
            CheckError(pgm_init_edge(e5_6 + i,
                        normalize_node[i],
                        classify_node[i], buf, &fast_mq_attr));

            sprintf(buf, "e6_7_%d", i);
            fast_mq_attr.nr_produce = sizeof(struct params_fine_collect_locations);
            fast_mq_attr.nr_consume = sizeof(struct params_fine_collect_locations);
            fast_mq_attr.nr_threshold = sizeof(struct params_fine_collect_locations);
            CheckError(pgm_init_edge(e6_7 + i,
                        classify_node[i],
                        collect_locations_node, buf, &fast_mq_attr));
        }

        fast_mq_attr.nr_produce = sizeof(struct params_display);
        fast_mq_attr.nr_consume = sizeof(struct params_display);
        fast_mq_attr.nr_threshold = sizeof(struct params_display);
        CheckError(pgm_init_edge(&e7_8, collect_locations_node, display_node, "e7_8", &fast_mq_attr));

        pthread_barrier_init(fine_init_barrier, 0, 3 * NUM_SCALE_LEVELS + 4);

        thread** t0  = arr_t0 + g_idx;
        thread** t1  = arr_t1 + g_idx;
        thread** t234 = arr_t234 + g_idx * NUM_SCALE_LEVELS;
        thread** t5  = arr_t5 + g_idx * NUM_SCALE_LEVELS;
        thread** t6  = arr_t6 + g_idx * NUM_SCALE_LEVELS;
        thread** t7  = arr_t7 + g_idx;
        thread** t8  = arr_t8 + g_idx;

        struct sync_info* in_sync_info_ABC                = arr_sync_info_ABC               [((g_idx + args.num_fine_graphs - 1) % args.num_fine_graphs)];
        struct sync_info* in_sync_info_normalize          = arr_sync_info_normalize         [((g_idx + args.num_fine_graphs - 1) % args.num_fine_graphs)];
        struct sync_info* in_sync_info_classify           = arr_sync_info_classify          [((g_idx + args.num_fine_graphs - 1) % args.num_fine_graphs)];

        struct sync_info* out_sync_info_ABC                = arr_sync_info_ABC               [g_idx];
        struct sync_info* out_sync_info_normalize          = arr_sync_info_normalize         [g_idx];
        struct sync_info* out_sync_info_classify           = arr_sync_info_classify          [g_idx];

        for (int i=0; i<NUM_SCALE_LEVELS; i++) {
            sync_info_init(i + in_sync_info_ABC);
            sync_info_init(i + in_sync_info_normalize);
            sync_info_init(i + in_sync_info_classify);

            sync_info_init(i + out_sync_info_ABC);
            sync_info_init(i + out_sync_info_normalize);
            sync_info_init(i + out_sync_info_classify);
        }

        /* G-FL fine-grained HOG */
        float bound_color_convert                   = 31.0194960215;
        float bound_compute_scales                  = 30.5688293615;
        // float bounds_resize      [NUM_SCALE_LEVELS] = {30.2419972231,30.3148010256,30.2936775569,30.3079650324,30.2963991483,30.3058228698,30.3005336032,30.2888197835,30.293491626,30.2840584057,30.3013304497,30.3016323015,30.3111116087};
        // float bounds_compute_grad[NUM_SCALE_LEVELS] = {30.4902863291,30.44552396,30.4706995604,30.4291495557,30.4171804989,30.4135009326,30.3785928987,30.4059076714,30.3816881109,30.3627509567,30.3564101342,30.3519205285,30.3405616838};
        // float bounds_compute_hist[NUM_SCALE_LEVELS] = {30.5326861275,30.4951636278,30.5415047381,30.4958614402,30.5149987735,30.4750125581,30.5373623676,30.522424222,30.4851937235,30.4577396346,30.4263855713,30.4118414514,30.5202006157};
        float bounds_ABC         [NUM_SCALE_LEVELS] = {30.2419972231,30.3148010256,30.2936775569,30.3079650324,30.2963991483,30.3058228698,30.3005336032,30.2888197835,30.293491626,30.2840584057,30.3013304497,30.3016323015,30.3111116087};
        float bounds_normalize   [NUM_SCALE_LEVELS] = {30.2903383654,30.3080869341,30.2928467608,30.3064130287,30.3114078316,30.3120987837,30.2985990076,30.2803770804,30.3099685825,30.3123524377,30.2989110618,30.2943410679,30.3081075149};
        float bounds_classify    [NUM_SCALE_LEVELS] = {30.4792191337,30.4948973087,30.4540269983,30.429525815,30.5229441951,30.5070612675,30.497249677,30.4955912512,30.4267037822,30.4508744413,30.430038752,30.3991297266,30.3767439685};
        //float bound_collect_locations               30.983976896;

        /* G-FL fine-grained HOG */
        float cost_color_convert                   = 4.509346;
        float cost_compute_scales                  = 1.947347;
        // float costs_resize      [NUM_SCALE_LEVELS] = {0.089336,0.503219,0.383134,0.464357,0.398606,0.452179,0.42211,0.355518,0.382077,0.32845,0.42664,0.428356,0.482245};
        // float costs_compute_grad[NUM_SCALE_LEVELS] = {1.500837,1.246367,1.389488,1.15328,1.085237,1.064319,0.86587,1.021152,0.883466,0.77581,0.739763,0.71424,0.649666};
        // float costs_compute_hist[NUM_SCALE_LEVELS] = {1.741876,1.528564,1.792009,1.532531,1.641325,1.414007,1.76846,1.683538,1.471886,1.315812,1.137567,1.054885,1.670897};
        float costs_ABC         [NUM_SCALE_LEVELS] = {0.089336,0.503219,0.383134,0.464357,0.398606,0.452179,0.42211,0.355518,0.382077,0.32845,0.42664,0.428356,0.482245};
        float costs_normalize   [NUM_SCALE_LEVELS] = {0.364151,0.46505,0.378411,0.455534,0.483929,0.487857,0.411112,0.307522,0.475747,0.489299,0.412886,0.386906,0.465167};
        float costs_classify    [NUM_SCALE_LEVELS] = {1.437921,1.52705,1.294706,1.155419,1.686494,1.596201,1.540423,1.530995,1.139376,1.276784,1.158335,0.98262,0.855359};
        float cost_collect_locations               = 4.307423;

        /* C-FL fine-grained HOG */
        //float bound_color_convert                   = 26.3496352098;
        //float bound_compute_scales                  = 25.6841982494;
        //float bounds_resize      [NUM_SCALE_LEVELS] = {25.0939476981,25.1387935171,25.130189875,25.1357033275,25.1312621102,25.1463332249,25.1316714257,25.1302245094,25.1267072576,25.1262401226,25.1348909935,25.1432006726,25.1413123823};
        //float bounds_compute_grad[NUM_SCALE_LEVELS] = {25.3489965402,25.4185896374,25.4092134479,25.4079932294,25.4051932818,25.4030496702,25.312479578,25.291207475,25.2868561354,25.2685316772,25.2518075571,25.246440943,25.2364255915};
        //float bounds_compute_hist[NUM_SCALE_LEVELS] = {25.5728014354,25.4973631396,25.5123077387,25.471628638,25.3841716342,25.3274278981,25.4720253593,25.5328196023,25.4654399579,25.3426074958,25.4860697502,25.3209827517,25.4610373822};
        //float bounds_ABC         [NUM_SCALE_LEVELS] = {25.0939476981,25.1387935171,25.130189875,25.1357033275,25.1312621102,25.1463332249,25.1316714257,25.1302245094,25.1267072576,25.1262401226,25.1348909935,25.1432006726,25.1413123823};
        //float bounds_normalize   [NUM_SCALE_LEVELS] = {25.1658996554,25.1786894798,25.1487118351,25.1680595823,25.1750299693,25.1885187788,25.1646279147,25.1649092834,25.1582577618,25.1765865135,25.1940356661,25.1748711091,25.179600851};
        //float bounds_classify    [NUM_SCALE_LEVELS] = {25.2909527262,25.3017792668,25.4537335318,25.3216511097,25.4693284561,25.3272781973,25.3334016159,25.3265829333,25.3061011243,25.2757711248,25.2811148401,25.2748319888,25.3276382806};
        ////float bound_collect_locations               26.204139258;

        /////* C-FL fine-grained HOG */
        //float cost_color_convert                   = 4.475663;
        //float cost_compute_scales                  = 2.150868;
        //float costs_resize      [NUM_SCALE_LEVELS] = {0.088747,0.245422,0.215364,0.234626,0.21911,0.271763,0.22054,0.215485,0.203197,0.201565,0.231788,0.260819,0.254222};
        //float costs_compute_grad[NUM_SCALE_LEVELS] = {0.979795,1.222928,1.190171,1.185908,1.176126,1.168637,0.852218,0.777901,0.762699,0.69868,0.640252,0.621503,0.586513};
        //float costs_compute_hist[NUM_SCALE_LEVELS] = {1.761688,1.498134,1.550345,1.408227,1.102684,0.904442,1.409613,1.622006,1.386606,0.957474,1.458679,0.881925,1.371225};
        //float costs_ABC         [NUM_SCALE_LEVELS] = {0.088747,0.245422,0.215364,0.234626,0.21911,0.271763,0.22054,0.215485,0.203197,0.201565,0.231788,0.260819,0.254222};
        //float costs_normalize   [NUM_SCALE_LEVELS] = {0.340121,0.384804,0.280073,0.347667,0.372019,0.419144,0.335678,0.336661,0.313423,0.377457,0.438418,0.371464,0.387988};
        //float costs_classify    [NUM_SCALE_LEVELS] = {0.777011,0.814835,1.345708,0.88426,1.400191,0.903919,0.925312,0.90149,0.829934,0.723972,0.742641,0.720691,0.905177};
        //float cost_collect_locations               = 3.967353;

        /* | first graph release      | second graph release     | first graph release again
         *  <---------PERIOD--------->
         *  <--------------- PERIOD * args.num_fine_graphs ---------->
         */
        int period = PERIOD * args.num_fine_graphs;
        int m_cpus = 16;
        struct task_info t_info;
        t_info.early = args.early;
        t_info.realtime = args.realtime;
        t_info.sched = fine_ABC;
        t_info.period = period;
        t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, cost_color_convert);
        t_info.phase = PERIOD * g_idx;
        t_info.id = 0;
        if (args.cluster != -1)
            t_info.cluster = args.cluster;
        else
            t_info.cluster = args.cluster;
        *t0 = new thread(&App::thread_color_convert, this,
                &color_convert_node, fine_init_barrier,
                gpu_hog, cpu_hog, frames, t_info, g_idx);

        t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, cost_compute_scales);
        t_info.id = 1;
        t_info.phase = t_info.phase + bound_color_convert;
        *t1 = new thread(&cv::cuda::HOG::thread_fine_compute_scales, gpu_hog,
                &compute_scales_node, fine_init_barrier, t_info);

        for (int i=0; i<NUM_SCALE_LEVELS; i++) {
            t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, costs_ABC[i]);
            t_info.id = 0 * NUM_SCALE_LEVELS + i + 2;
            t_info.phase = PERIOD * g_idx + bound_color_convert + bound_compute_scales;
            t_info.s_info_in = in_sync_info_ABC + i;
            t_info.s_info_out = out_sync_info_ABC + i;
            t234[i] = new thread(&cv::cuda::HOG::thread_fine_ABC, gpu_hog,
                    ABC_node + i, fine_init_barrier, t_info);

            t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, costs_normalize[i]);
            t_info.id = 1 * NUM_SCALE_LEVELS + i + 2;
            t_info.phase = t_info.phase + bounds_ABC[i];
            t_info.s_info_in = in_sync_info_normalize + i;
            t_info.s_info_out = out_sync_info_normalize + i;
            t5[i] = new thread(&cv::cuda::HOG::thread_fine_normalize_histograms,
                    gpu_hog, normalize_node + i, fine_init_barrier, t_info);

            t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, costs_classify[i]);
            t_info.id = 2 * NUM_SCALE_LEVELS + i + 2;
            t_info.phase = t_info.phase + bounds_normalize[i];
            t_info.s_info_in = in_sync_info_classify + i;
            t_info.s_info_out = out_sync_info_classify + i;
            t6[i] = new thread(&cv::cuda::HOG::thread_fine_classify, gpu_hog,
                    classify_node + i, fine_init_barrier, t_info);
        }

        t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, cost_collect_locations);
        t_info.id = 3 * NUM_SCALE_LEVELS + 2;
        t_info.phase = t_info.phase + *std::max_element(bounds_classify, bounds_classify+NUM_SCALE_LEVELS);
        *t7 = new thread(&cv::cuda::HOG::thread_fine_collect_locations, gpu_hog,
                &collect_locations_node, fine_init_barrier, t_info);
        //t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, cost_display);
        //t_info.phase;
        *t8 = new thread(&App::thread_display, this, &display_node,
                fine_init_barrier, g_idx == 0 && args.display);
    }

    /* graph construction finishes */

    printf("Joining pthreads...\n");

    for (int g_idx = 0; g_idx < args.num_fine_graphs; g_idx++) {
        graph_t g = arr_g[g_idx];
        thread* t0 = arr_t0[g_idx];
        thread* t1 = arr_t1[g_idx];
        thread** t234 = arr_t234 + g_idx * NUM_SCALE_LEVELS;
        thread** t5  = arr_t5 + g_idx * NUM_SCALE_LEVELS;
        thread** t6  = arr_t6 + g_idx * NUM_SCALE_LEVELS;
        thread* t7 = arr_t7[g_idx];
        thread* t8 = arr_t8[g_idx];
        t0->join();
        t1->join();
        delete t0;
        delete t1;
        for (int i=0; i<NUM_SCALE_LEVELS; i++) {
            if (t234[i]->joinable()) t234[i]->join();
            if (t5[i]->joinable()) t5[i]->join();
            if (t6[i]->joinable()) t6[i]->join();

            delete t234[i];
            delete t5[i];
            delete t6[i];
        }
        t7->join();
        t8->join();
        delete t7;
        delete t8;
        CheckError(pgm_destroy_graph(g));
    }
    free(arr_t1);
    free(arr_t234);
    free(arr_t5);
    free(arr_t6);
    free(arr_t7);
    free(arr_t8);

    //CheckError(pgm_destroy_graph(g));
    CheckError(pgm_destroy());
    fprintf(stdout, "cleaned up ...");
}

void App::sched_BCD_hog(cv::Ptr<cv::cuda::HOG> gpu_hog, cv::HOGDescriptor cpu_hog, Mat* frames)
{
    pthread_barrier_t arr_fine_init_barrier[args.num_fine_graphs];
    /* graph construction */
    graph_t arr_g [args.num_fine_graphs];

    struct sync_info arr_sync_info_resize   [args.num_fine_graphs][NUM_SCALE_LEVELS];
    struct sync_info arr_sync_info_BCD      [args.num_fine_graphs][NUM_SCALE_LEVELS];
    struct sync_info arr_sync_info_classify [args.num_fine_graphs][NUM_SCALE_LEVELS];

    thread** arr_t0   = (thread**) calloc(args.num_fine_graphs, sizeof(std::thread *));
    thread** arr_t1   = (thread**) calloc(args.num_fine_graphs, sizeof(std::thread *));
    thread** arr_t2   = (thread**) calloc(args.num_fine_graphs * NUM_SCALE_LEVELS, sizeof(std::thread *));
    thread** arr_t345 = (thread**) calloc(args.num_fine_graphs * NUM_SCALE_LEVELS, sizeof(std::thread *));
    thread** arr_t6   = (thread**) calloc(args.num_fine_graphs * NUM_SCALE_LEVELS, sizeof(std::thread *));
    thread** arr_t7   = (thread**) calloc(args.num_fine_graphs, sizeof(std::thread *));
    thread** arr_t8   = (thread**) calloc(args.num_fine_graphs, sizeof(std::thread *));

    char buf[30];
    sprintf(buf, "/tmp/graph_t%d", args.task_id);
    CheckError(pgm_init(buf, 1));

    for (int g_idx = 0; g_idx < args.num_fine_graphs; g_idx++)
    {
        pthread_barrier_t* fine_init_barrier = arr_fine_init_barrier + g_idx;

        // A graph consists of the graph itself, nodes, and edge
        graph_t* g_ptr = arr_g + g_idx;

        node_t color_convert_node;
        node_t compute_scales_node;
        node_t resize_node             [NUM_SCALE_LEVELS];
        node_t BCD_node                [NUM_SCALE_LEVELS]; // compute-grads -> normalize-hists
        node_t classify_node           [NUM_SCALE_LEVELS];
        node_t collect_locations_node;
        node_t display_node;

        edge_t e0_1;
        edge_t e1_2   [NUM_SCALE_LEVELS];
        edge_t e2_345 [NUM_SCALE_LEVELS];
        edge_t e345_6 [NUM_SCALE_LEVELS];
        edge_t e6_7   [NUM_SCALE_LEVELS];
        edge_t e7_8;

        // Initialize the graph
        sprintf(buf, "hog_%d", g_idx);
        CheckError(pgm_init_graph(g_ptr, buf));
        graph_t g = *g_ptr;

        // Initialize the nodes
        CheckError(pgm_init_node(&color_convert_node, g, "color_convert"));
        CheckError(pgm_init_node(&compute_scales_node, g, "compute_scales"));
        for (int i=0; i<NUM_SCALE_LEVELS; i++) {
            CheckError(pgm_init_node(resize_node + i, g, "resize"));
            CheckError(pgm_init_node(BCD_node + i, g, "BCD"));
            CheckError(pgm_init_node(classify_node + i, g, "classify"));
        }
        CheckError(pgm_init_node(&collect_locations_node, g, "collect_locations"));
        CheckError(pgm_init_node(&display_node, g, "display"));

        // Initialize the edges
        edge_attr_t fast_mq_attr;
        memset(&fast_mq_attr, 0, sizeof(fast_mq_attr));
        fast_mq_attr.type = pgm_fast_fifo_edge;

        fast_mq_attr.nr_produce = sizeof(struct params_compute);
        fast_mq_attr.nr_consume = sizeof(struct params_compute);
        fast_mq_attr.nr_threshold = sizeof(struct params_compute);
        CheckError(pgm_init_edge(&e0_1, color_convert_node, compute_scales_node, "e0_1", &fast_mq_attr));

        for (int i=0; i<NUM_SCALE_LEVELS; i++) {
            sprintf(buf, "e1_2_%d", i);
            fast_mq_attr.nr_produce = sizeof(struct params_resize);
            fast_mq_attr.nr_consume = sizeof(struct params_resize);
            fast_mq_attr.nr_threshold = sizeof(struct params_resize);
            CheckError(pgm_init_edge(e1_2 + i,
                        compute_scales_node,
                        resize_node[i], buf, &fast_mq_attr));

            sprintf(buf, "e2_345_%d", i);
            fast_mq_attr.nr_produce = sizeof(struct params_compute_gradients);
            fast_mq_attr.nr_consume = sizeof(struct params_compute_gradients);
            fast_mq_attr.nr_threshold = sizeof(struct params_compute_gradients);
            CheckError(pgm_init_edge(e2_345 + i,
                        resize_node[i],
                        BCD_node[i], buf, &fast_mq_attr));

            sprintf(buf, "e345_6_%d", i);
            fast_mq_attr.nr_produce = sizeof(struct params_fine_classify);
            fast_mq_attr.nr_consume = sizeof(struct params_fine_classify);
            fast_mq_attr.nr_threshold = sizeof(struct params_fine_classify);
            CheckError(pgm_init_edge(e345_6 + i,
                        BCD_node[i],
                        classify_node[i], buf, &fast_mq_attr));

            sprintf(buf, "e6_7_%d", i);
            fast_mq_attr.nr_produce = sizeof(struct params_fine_collect_locations);
            fast_mq_attr.nr_consume = sizeof(struct params_fine_collect_locations);
            fast_mq_attr.nr_threshold = sizeof(struct params_fine_collect_locations);
            CheckError(pgm_init_edge(e6_7 + i,
                        classify_node[i],
                        collect_locations_node, buf, &fast_mq_attr));
        }

        fast_mq_attr.nr_produce = sizeof(struct params_display);
        fast_mq_attr.nr_consume = sizeof(struct params_display);
        fast_mq_attr.nr_threshold = sizeof(struct params_display);
        CheckError(pgm_init_edge(&e7_8, collect_locations_node, display_node, "e7_8", &fast_mq_attr));

        pthread_barrier_init(fine_init_barrier, 0, 3 * NUM_SCALE_LEVELS + 4);

        thread** t0   = arr_t0 + g_idx;
        thread** t1   = arr_t1 + g_idx;
        thread** t2   = arr_t2 + g_idx * NUM_SCALE_LEVELS;
        thread** t345 = arr_t345 + g_idx * NUM_SCALE_LEVELS;
        thread** t6   = arr_t6 + g_idx * NUM_SCALE_LEVELS;
        thread** t7   = arr_t7 + g_idx;
        thread** t8   = arr_t8 + g_idx;

        struct sync_info* in_sync_info_resize             = arr_sync_info_resize            [((g_idx + args.num_fine_graphs - 1) % args.num_fine_graphs)];
        struct sync_info* in_sync_info_BCD                = arr_sync_info_BCD               [((g_idx + args.num_fine_graphs - 1) % args.num_fine_graphs)];
        struct sync_info* in_sync_info_classify           = arr_sync_info_classify          [((g_idx + args.num_fine_graphs - 1) % args.num_fine_graphs)];

        struct sync_info* out_sync_info_resize             = arr_sync_info_resize            [g_idx];
        struct sync_info* out_sync_info_BCD                = arr_sync_info_BCD               [g_idx];
        struct sync_info* out_sync_info_classify           = arr_sync_info_classify          [g_idx];

        for (int i=0; i<NUM_SCALE_LEVELS; i++) {
            sync_info_init(i + in_sync_info_resize);
            sync_info_init(i + in_sync_info_BCD);
            sync_info_init(i + in_sync_info_classify);

            sync_info_init(i + out_sync_info_resize);
            sync_info_init(i + out_sync_info_BCD);
            sync_info_init(i + out_sync_info_classify);
        }

        /* G-FL fine-grained HOG */
        float bound_color_convert                   = 31.0194960215;
        float bound_compute_scales                  = 30.5688293615;
        float bounds_resize      [NUM_SCALE_LEVELS] = {30.2419972231,30.3148010256,30.2936775569,30.3079650324,30.2963991483,30.3058228698,30.3005336032,30.2888197835,30.293491626,30.2840584057,30.3013304497,30.3016323015,30.3111116087};
        // float bounds_compute_grad[NUM_SCALE_LEVELS] = {30.4902863291,30.44552396,30.4706995604,30.4291495557,30.4171804989,30.4135009326,30.3785928987,30.4059076714,30.3816881109,30.3627509567,30.3564101342,30.3519205285,30.3405616838};
        // float bounds_compute_hist[NUM_SCALE_LEVELS] = {30.5326861275,30.4951636278,30.5415047381,30.4958614402,30.5149987735,30.4750125581,30.5373623676,30.522424222,30.4851937235,30.4577396346,30.4263855713,30.4118414514,30.5202006157};
        float bounds_BCD         [NUM_SCALE_LEVELS] = {30.4902863291,30.44552396,30.4706995604,30.4291495557,30.4171804989,30.4135009326,30.3785928987,30.4059076714,30.3816881109,30.3627509567,30.3564101342,30.3519205285,30.3405616838};
        // float bounds_normalize   [NUM_SCALE_LEVELS] = {30.2903383654,30.3080869341,30.2928467608,30.3064130287,30.3114078316,30.3120987837,30.2985990076,30.2803770804,30.3099685825,30.3123524377,30.2989110618,30.2943410679,30.3081075149};
        float bounds_classify    [NUM_SCALE_LEVELS] = {30.4792191337,30.4948973087,30.4540269983,30.429525815,30.5229441951,30.5070612675,30.497249677,30.4955912512,30.4267037822,30.4508744413,30.430038752,30.3991297266,30.3767439685};
        //float bound_collect_locations               30.983976896;

        /* G-FL fine-grained HOG */
        float cost_color_convert                   = 4.509346;
        float cost_compute_scales                  = 1.947347;
        float costs_resize      [NUM_SCALE_LEVELS] = {0.089336,0.503219,0.383134,0.464357,0.398606,0.452179,0.42211,0.355518,0.382077,0.32845,0.42664,0.428356,0.482245};
        // float costs_compute_grad[NUM_SCALE_LEVELS] = {1.500837,1.246367,1.389488,1.15328,1.085237,1.064319,0.86587,1.021152,0.883466,0.77581,0.739763,0.71424,0.649666};
        // float costs_compute_hist[NUM_SCALE_LEVELS] = {1.741876,1.528564,1.792009,1.532531,1.641325,1.414007,1.76846,1.683538,1.471886,1.315812,1.137567,1.054885,1.670897};
        float costs_BCD         [NUM_SCALE_LEVELS] = {1.500837,1.246367,1.389488,1.15328,1.085237,1.064319,0.86587,1.021152,0.883466,0.77581,0.739763,0.71424,0.649666};
        // float costs_normalize   [NUM_SCALE_LEVELS] = {0.364151,0.46505,0.378411,0.455534,0.483929,0.487857,0.411112,0.307522,0.475747,0.489299,0.412886,0.386906,0.465167};
        float costs_classify    [NUM_SCALE_LEVELS] = {1.437921,1.52705,1.294706,1.155419,1.686494,1.596201,1.540423,1.530995,1.139376,1.276784,1.158335,0.98262,0.855359};
        float cost_collect_locations               = 4.307423;

        /* C-FL fine-grained HOG */
        //float bound_color_convert                   = 26.3496352098;
        //float bound_compute_scales                  = 25.6841982494;
        //float bounds_resize      [NUM_SCALE_LEVELS] = {25.0939476981,25.1387935171,25.130189875,25.1357033275,25.1312621102,25.1463332249,25.1316714257,25.1302245094,25.1267072576,25.1262401226,25.1348909935,25.1432006726,25.1413123823};
        //float bounds_compute_grad[NUM_SCALE_LEVELS] = {25.3489965402,25.4185896374,25.4092134479,25.4079932294,25.4051932818,25.4030496702,25.312479578,25.291207475,25.2868561354,25.2685316772,25.2518075571,25.246440943,25.2364255915};
        //float bounds_BCD         [NUM_SCALE_LEVELS] = {25.3489965402,25.4185896374,25.4092134479,25.4079932294,25.4051932818,25.4030496702,25.312479578,25.291207475,25.2868561354,25.2685316772,25.2518075571,25.246440943,25.2364255915};
        //float bounds_compute_hist[NUM_SCALE_LEVELS] = {25.5728014354,25.4973631396,25.5123077387,25.471628638,25.3841716342,25.3274278981,25.4720253593,25.5328196023,25.4654399579,25.3426074958,25.4860697502,25.3209827517,25.4610373822};
        //float bounds_normalize   [NUM_SCALE_LEVELS] = {25.1658996554,25.1786894798,25.1487118351,25.1680595823,25.1750299693,25.1885187788,25.1646279147,25.1649092834,25.1582577618,25.1765865135,25.1940356661,25.1748711091,25.179600851};
        //float bounds_classify    [NUM_SCALE_LEVELS] = {25.2909527262,25.3017792668,25.4537335318,25.3216511097,25.4693284561,25.3272781973,25.3334016159,25.3265829333,25.3061011243,25.2757711248,25.2811148401,25.2748319888,25.3276382806};
        ////float bound_collect_locations               26.204139258;

        /////* C-FL fine-grained HOG */
        //float cost_color_convert                   = 4.475663;
        //float cost_compute_scales                  = 2.150868;
        //float costs_resize      [NUM_SCALE_LEVELS] = {0.088747,0.245422,0.215364,0.234626,0.21911,0.271763,0.22054,0.215485,0.203197,0.201565,0.231788,0.260819,0.254222};
        //float costs_compute_grad[NUM_SCALE_LEVELS] = {0.979795,1.222928,1.190171,1.185908,1.176126,1.168637,0.852218,0.777901,0.762699,0.69868,0.640252,0.621503,0.586513};
        //float costs_BCD         [NUM_SCALE_LEVELS] = {0.979795,1.222928,1.190171,1.185908,1.176126,1.168637,0.852218,0.777901,0.762699,0.69868,0.640252,0.621503,0.586513};
        //float costs_compute_hist[NUM_SCALE_LEVELS] = {1.761688,1.498134,1.550345,1.408227,1.102684,0.904442,1.409613,1.622006,1.386606,0.957474,1.458679,0.881925,1.371225};
        //float costs_normalize   [NUM_SCALE_LEVELS] = {0.340121,0.384804,0.280073,0.347667,0.372019,0.419144,0.335678,0.336661,0.313423,0.377457,0.438418,0.371464,0.387988};
        //float costs_classify    [NUM_SCALE_LEVELS] = {0.777011,0.814835,1.345708,0.88426,1.400191,0.903919,0.925312,0.90149,0.829934,0.723972,0.742641,0.720691,0.905177};
        //float cost_collect_locations               = 3.967353;

        /* | first graph release      | second graph release     | first graph release again
         *  <---------PERIOD--------->
         *  <--------------- PERIOD * args.num_fine_graphs ---------->
         */
        int period = PERIOD * args.num_fine_graphs;
        int m_cpus = 16;
        struct task_info t_info;
        t_info.early = args.early;
        t_info.realtime = args.realtime;
        t_info.sched = fine_BCD;
        t_info.period = period;
        t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, cost_color_convert);
        t_info.phase = PERIOD * g_idx;
        t_info.id = 0;
        if (args.cluster != -1)
            t_info.cluster = args.cluster;
        else
            t_info.cluster = args.cluster;
        *t0 = new thread(&App::thread_color_convert, this,
                &color_convert_node, fine_init_barrier,
                gpu_hog, cpu_hog, frames, t_info, g_idx);

        t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, cost_compute_scales);
        t_info.id = 1;
        t_info.phase = t_info.phase + bound_color_convert;
        *t1 = new thread(&cv::cuda::HOG::thread_fine_compute_scales, gpu_hog,
                &compute_scales_node, fine_init_barrier, t_info);

        for (int i=0; i<NUM_SCALE_LEVELS; i++) {
            t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, costs_resize[i]);
            t_info.id = 0 * NUM_SCALE_LEVELS + i + 2;
            t_info.phase = PERIOD * g_idx + bound_color_convert + bound_compute_scales;
            t_info.s_info_in = in_sync_info_resize + i;
            t_info.s_info_out = out_sync_info_resize + i;
            t2[i] = new thread(&cv::cuda::HOG::thread_fine_resize, gpu_hog,
                    resize_node + i, fine_init_barrier, t_info);

            t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, costs_BCD[i]);
            t_info.id = 1 * NUM_SCALE_LEVELS + i + 2;
            t_info.phase = PERIOD * g_idx + bounds_resize[i];
            t_info.s_info_in = in_sync_info_BCD + i;
            t_info.s_info_out = out_sync_info_BCD + i;
            t345[i] = new thread(&cv::cuda::HOG::thread_fine_BCD, gpu_hog,
                    BCD_node + i, fine_init_barrier, t_info);

            t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, costs_classify[i]);
            t_info.id = 2 * NUM_SCALE_LEVELS + i + 2;
            t_info.phase = t_info.phase + bounds_BCD[i];
            t_info.s_info_in = in_sync_info_classify + i;
            t_info.s_info_out = out_sync_info_classify + i;
            t6[i] = new thread(&cv::cuda::HOG::thread_fine_classify, gpu_hog,
                    classify_node + i, fine_init_barrier, t_info);
        }

        t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, cost_collect_locations);
        t_info.id = 3 * NUM_SCALE_LEVELS + 2;
        t_info.phase = t_info.phase + *std::max_element(bounds_classify, bounds_classify+NUM_SCALE_LEVELS);
        *t7 = new thread(&cv::cuda::HOG::thread_fine_collect_locations, gpu_hog,
                &collect_locations_node, fine_init_barrier, t_info);
        //t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, cost_display);
        //t_info.phase;
        *t8 = new thread(&App::thread_display, this, &display_node,
                fine_init_barrier, g_idx == 0 && args.display);
    }

    /* graph construction finishes */

    printf("Joining pthreads...\n");

    for (int g_idx = 0; g_idx < args.num_fine_graphs; g_idx++) {
        graph_t g = arr_g[g_idx];
        thread* t0 = arr_t0[g_idx];
        thread* t1 = arr_t1[g_idx];
        thread** t2 = arr_t2 + g_idx * NUM_SCALE_LEVELS;
        thread** t345 = arr_t345 + g_idx * NUM_SCALE_LEVELS;
        thread** t6  = arr_t6 + g_idx * NUM_SCALE_LEVELS;
        thread* t7 = arr_t7[g_idx];
        thread* t8 = arr_t8[g_idx];
        t0->join();
        t1->join();
        delete t0;
        delete t1;
        for (int i=0; i<NUM_SCALE_LEVELS; i++) {
            if (t2[i]->joinable()) t2[i]->join();
            if (t345[i]->joinable()) t345[i]->join();
            if (t6[i]->joinable()) t6[i]->join();

            delete t2[i];
            delete t345[i];
            delete t6[i];
        }
        t7->join();
        t8->join();
        delete t7;
        delete t8;
        CheckError(pgm_destroy_graph(g));
    }
    free(arr_t1);
    free(arr_t2);
    free(arr_t345);
    free(arr_t6);
    free(arr_t7);
    free(arr_t8);

    //CheckError(pgm_destroy_graph(g));
    CheckError(pgm_destroy());
    fprintf(stdout, "cleaned up ...");
}

void App::sched_CDE_hog(cv::Ptr<cv::cuda::HOG> gpu_hog, cv::HOGDescriptor cpu_hog, Mat* frames)
{
    pthread_barrier_t arr_fine_init_barrier[args.num_fine_graphs];
    /* graph construction */
    graph_t arr_g [args.num_fine_graphs];

    struct sync_info arr_sync_info_resize            [args.num_fine_graphs][NUM_SCALE_LEVELS];
    struct sync_info arr_sync_info_compute_gradients [args.num_fine_graphs][NUM_SCALE_LEVELS];
    struct sync_info arr_sync_info_CDE               [args.num_fine_graphs][NUM_SCALE_LEVELS];

    thread** arr_t0   = (thread**) calloc(args.num_fine_graphs, sizeof(std::thread *));
    thread** arr_t1   = (thread**) calloc(args.num_fine_graphs, sizeof(std::thread *));
    thread** arr_t2   = (thread**) calloc(args.num_fine_graphs * NUM_SCALE_LEVELS, sizeof(std::thread *));
    thread** arr_t3   = (thread**) calloc(args.num_fine_graphs * NUM_SCALE_LEVELS, sizeof(std::thread *));
    thread** arr_t456 = (thread**) calloc(args.num_fine_graphs * NUM_SCALE_LEVELS, sizeof(std::thread *));
    thread** arr_t7   = (thread**) calloc(args.num_fine_graphs, sizeof(std::thread *));
    thread** arr_t8   = (thread**) calloc(args.num_fine_graphs, sizeof(std::thread *));

    char buf[30];
    sprintf(buf, "/tmp/graph_t%d", args.task_id);
    CheckError(pgm_init(buf, 1));

    for (int g_idx = 0; g_idx < args.num_fine_graphs; g_idx++)
    {
        pthread_barrier_t* fine_init_barrier = arr_fine_init_barrier + g_idx;

        // A graph consists of the graph itself, nodes, and edge
        graph_t* g_ptr = arr_g + g_idx;

        node_t color_convert_node;
        node_t compute_scales_node;
        node_t resize_node            [NUM_SCALE_LEVELS];
        node_t compute_gradients_node [NUM_SCALE_LEVELS];
        node_t CDE_node               [NUM_SCALE_LEVELS]; // compute-hists -> classify-hists
        node_t collect_locations_node;
        node_t display_node;

        edge_t e0_1;
        edge_t e1_2   [NUM_SCALE_LEVELS];
        edge_t e2_3   [NUM_SCALE_LEVELS];
        edge_t e3_456 [NUM_SCALE_LEVELS];
        edge_t e456_7 [NUM_SCALE_LEVELS];
        edge_t e7_8;

        // Initialize the graph
        sprintf(buf, "hog_%d", g_idx);
        CheckError(pgm_init_graph(g_ptr, buf));
        graph_t g = *g_ptr;

        // Initialize the nodes
        CheckError(pgm_init_node(&color_convert_node, g, "color_convert"));
        CheckError(pgm_init_node(&compute_scales_node, g, "compute_scales"));
        for (int i=0; i<NUM_SCALE_LEVELS; i++) {
            CheckError(pgm_init_node(resize_node + i, g, "resize"));
            CheckError(pgm_init_node(compute_gradients_node + i, g, "compute_gradients"));
            CheckError(pgm_init_node(CDE_node + i, g, "CDE"));
        }
        CheckError(pgm_init_node(&collect_locations_node, g, "collect_locations"));
        CheckError(pgm_init_node(&display_node, g, "display"));

        // Initialize the edges
        edge_attr_t fast_mq_attr;
        memset(&fast_mq_attr, 0, sizeof(fast_mq_attr));
        fast_mq_attr.type = pgm_fast_fifo_edge;

        fast_mq_attr.nr_produce = sizeof(struct params_compute);
        fast_mq_attr.nr_consume = sizeof(struct params_compute);
        fast_mq_attr.nr_threshold = sizeof(struct params_compute);
        CheckError(pgm_init_edge(&e0_1, color_convert_node, compute_scales_node, "e0_1", &fast_mq_attr));

        for (int i=0; i<NUM_SCALE_LEVELS; i++) {
            sprintf(buf, "e1_2_%d", i);
            fast_mq_attr.nr_produce = sizeof(struct params_resize);
            fast_mq_attr.nr_consume = sizeof(struct params_resize);
            fast_mq_attr.nr_threshold = sizeof(struct params_resize);
            CheckError(pgm_init_edge(e1_2 + i,
                        compute_scales_node,
                        resize_node[i], buf, &fast_mq_attr));

            sprintf(buf, "e2_3_%d", i);
            fast_mq_attr.nr_produce = sizeof(struct params_compute_gradients);
            fast_mq_attr.nr_consume = sizeof(struct params_compute_gradients);
            fast_mq_attr.nr_threshold = sizeof(struct params_compute_gradients);
            CheckError(pgm_init_edge(e2_3 + i,
                        resize_node[i],
                        compute_gradients_node[i], buf, &fast_mq_attr));

            sprintf(buf, "e3_456_%d", i);
            fast_mq_attr.nr_produce = sizeof(struct params_compute_histograms);
            fast_mq_attr.nr_consume = sizeof(struct params_compute_histograms);
            fast_mq_attr.nr_threshold = sizeof(struct params_compute_histograms);
            CheckError(pgm_init_edge(e3_456 + i,
                        compute_gradients_node[i],
                        CDE_node[i], buf, &fast_mq_attr));

            sprintf(buf, "e456_7_%d", i);
            fast_mq_attr.nr_produce = sizeof(struct params_fine_collect_locations);
            fast_mq_attr.nr_consume = sizeof(struct params_fine_collect_locations);
            fast_mq_attr.nr_threshold = sizeof(struct params_fine_collect_locations);
            CheckError(pgm_init_edge(e456_7 + i,
                        CDE_node[i],
                        collect_locations_node, buf, &fast_mq_attr));
        }

        fast_mq_attr.nr_produce = sizeof(struct params_display);
        fast_mq_attr.nr_consume = sizeof(struct params_display);
        fast_mq_attr.nr_threshold = sizeof(struct params_display);
        CheckError(pgm_init_edge(&e7_8, collect_locations_node, display_node, "e7_8", &fast_mq_attr));

        pthread_barrier_init(fine_init_barrier, 0, 3 * NUM_SCALE_LEVELS + 4);

        thread** t0  = arr_t0 + g_idx;
        thread** t1  = arr_t1 + g_idx;
        thread** t2  = arr_t2 + g_idx * NUM_SCALE_LEVELS;
        thread** t3  = arr_t3 + g_idx * NUM_SCALE_LEVELS;
        thread** t456 = arr_t456 + g_idx * NUM_SCALE_LEVELS;
        thread** t7  = arr_t7 + g_idx;
        thread** t8  = arr_t8 + g_idx;

        struct sync_info* in_sync_info_resize             = arr_sync_info_resize            [((g_idx + args.num_fine_graphs - 1) % args.num_fine_graphs)];
        struct sync_info* in_sync_info_compute_gradients  = arr_sync_info_compute_gradients [((g_idx + args.num_fine_graphs - 1) % args.num_fine_graphs)];
        struct sync_info* in_sync_info_CDE                = arr_sync_info_CDE               [((g_idx + args.num_fine_graphs - 1) % args.num_fine_graphs)];

        struct sync_info* out_sync_info_resize             = arr_sync_info_resize            [g_idx];
        struct sync_info* out_sync_info_compute_gradients  = arr_sync_info_compute_gradients [g_idx];
        struct sync_info* out_sync_info_CDE                = arr_sync_info_CDE               [g_idx];

        for (int i=0; i<NUM_SCALE_LEVELS; i++) {
            sync_info_init(i + in_sync_info_resize);
            sync_info_init(i + in_sync_info_compute_gradients);
            sync_info_init(i + in_sync_info_CDE);

            sync_info_init(i + out_sync_info_resize);
            sync_info_init(i + out_sync_info_compute_gradients);
            sync_info_init(i + out_sync_info_CDE);
        }

        /* G-FL fine-grained HOG */
        float bound_color_convert                   = 31.0194960215;
        float bound_compute_scales                  = 30.5688293615;
        float bounds_resize      [NUM_SCALE_LEVELS] = {30.2419972231,30.3148010256,30.2936775569,30.3079650324,30.2963991483,30.3058228698,30.3005336032,30.2888197835,30.293491626,30.2840584057,30.3013304497,30.3016323015,30.3111116087};
        float bounds_compute_grad[NUM_SCALE_LEVELS] = {30.4902863291,30.44552396,30.4706995604,30.4291495557,30.4171804989,30.4135009326,30.3785928987,30.4059076714,30.3816881109,30.3627509567,30.3564101342,30.3519205285,30.3405616838};
        // float bounds_compute_hist[NUM_SCALE_LEVELS] = {30.5326861275,30.4951636278,30.5415047381,30.4958614402,30.5149987735,30.4750125581,30.5373623676,30.522424222,30.4851937235,30.4577396346,30.4263855713,30.4118414514,30.5202006157};
        float bounds_CDE         [NUM_SCALE_LEVELS] = {30.5326861275,30.4951636278,30.5415047381,30.4958614402,30.5149987735,30.4750125581,30.5373623676,30.522424222,30.4851937235,30.4577396346,30.4263855713,30.4118414514,30.5202006157};
        // float bounds_normalize   [NUM_SCALE_LEVELS] = {30.2903383654,30.3080869341,30.2928467608,30.3064130287,30.3114078316,30.3120987837,30.2985990076,30.2803770804,30.3099685825,30.3123524377,30.2989110618,30.2943410679,30.3081075149};
        // float bounds_classify    [NUM_SCALE_LEVELS] = {30.4792191337,30.4948973087,30.4540269983,30.429525815,30.5229441951,30.5070612675,30.497249677,30.4955912512,30.4267037822,30.4508744413,30.430038752,30.3991297266,30.3767439685};
        //float bound_collect_locations               30.983976896;

        /* G-FL fine-grained HOG */
        float cost_color_convert                   = 4.509346;
        float cost_compute_scales                  = 1.947347;
        float costs_resize      [NUM_SCALE_LEVELS] = {0.089336,0.503219,0.383134,0.464357,0.398606,0.452179,0.42211,0.355518,0.382077,0.32845,0.42664,0.428356,0.482245};
        float costs_compute_grad[NUM_SCALE_LEVELS] = {1.500837,1.246367,1.389488,1.15328,1.085237,1.064319,0.86587,1.021152,0.883466,0.77581,0.739763,0.71424,0.649666};
        // float costs_compute_hist[NUM_SCALE_LEVELS] = {1.741876,1.528564,1.792009,1.532531,1.641325,1.414007,1.76846,1.683538,1.471886,1.315812,1.137567,1.054885,1.670897};
        float costs_CDE         [NUM_SCALE_LEVELS] = {1.741876,1.528564,1.792009,1.532531,1.641325,1.414007,1.76846,1.683538,1.471886,1.315812,1.137567,1.054885,1.670897};
        // float costs_normalize   [NUM_SCALE_LEVELS] = {0.364151,0.46505,0.378411,0.455534,0.483929,0.487857,0.411112,0.307522,0.475747,0.489299,0.412886,0.386906,0.465167};
        // float costs_classify    [NUM_SCALE_LEVELS] = {1.437921,1.52705,1.294706,1.155419,1.686494,1.596201,1.540423,1.530995,1.139376,1.276784,1.158335,0.98262,0.855359};
        float cost_collect_locations               = 4.307423;

        /* C-FL fine-grained HOG */
        //float bound_color_convert                   = 26.3496352098;
        //float bound_compute_scales                  = 25.6841982494;
        //float bounds_resize      [NUM_SCALE_LEVELS] = {25.0939476981,25.1387935171,25.130189875,25.1357033275,25.1312621102,25.1463332249,25.1316714257,25.1302245094,25.1267072576,25.1262401226,25.1348909935,25.1432006726,25.1413123823};
        //float bounds_compute_grad[NUM_SCALE_LEVELS] = {25.3489965402,25.4185896374,25.4092134479,25.4079932294,25.4051932818,25.4030496702,25.312479578,25.291207475,25.2868561354,25.2685316772,25.2518075571,25.246440943,25.2364255915};
        //float bounds_compute_hist[NUM_SCALE_LEVELS] = {25.5728014354,25.4973631396,25.5123077387,25.471628638,25.3841716342,25.3274278981,25.4720253593,25.5328196023,25.4654399579,25.3426074958,25.4860697502,25.3209827517,25.4610373822};
        // float bounds_CDE        [NUM_SCALE_LEVELS] = {25.5728014354,25.4973631396,25.5123077387,25.471628638,25.3841716342,25.3274278981,25.4720253593,25.5328196023,25.4654399579,25.3426074958,25.4860697502,25.3209827517,25.4610373822};
        //float bounds_normalize   [NUM_SCALE_LEVELS] = {25.1658996554,25.1786894798,25.1487118351,25.1680595823,25.1750299693,25.1885187788,25.1646279147,25.1649092834,25.1582577618,25.1765865135,25.1940356661,25.1748711091,25.179600851};
        //float bounds_classify    [NUM_SCALE_LEVELS] = {25.2909527262,25.3017792668,25.4537335318,25.3216511097,25.4693284561,25.3272781973,25.3334016159,25.3265829333,25.3061011243,25.2757711248,25.2811148401,25.2748319888,25.3276382806};
        ////float bound_collect_locations               26.204139258;

        /////* C-FL fine-grained HOG */
        //float cost_color_convert                   = 4.475663;
        //float cost_compute_scales                  = 2.150868;
        //float costs_resize      [NUM_SCALE_LEVELS] = {0.088747,0.245422,0.215364,0.234626,0.21911,0.271763,0.22054,0.215485,0.203197,0.201565,0.231788,0.260819,0.254222};
        //float costs_compute_grad[NUM_SCALE_LEVELS] = {0.979795,1.222928,1.190171,1.185908,1.176126,1.168637,0.852218,0.777901,0.762699,0.69868,0.640252,0.621503,0.586513};
        // float costs_compute_hist[NUM_SCALE_LEVELS] = {1.761688,1.498134,1.550345,1.408227,1.102684,0.904442,1.409613,1.622006,1.386606,0.957474,1.458679,0.881925,1.371225};
        //float costs_CDE         [NUM_SCALE_LEVELS] = {1.761688,1.498134,1.550345,1.408227,1.102684,0.904442,1.409613,1.622006,1.386606,0.957474,1.458679,0.881925,1.371225};
        //float costs_normalize   [NUM_SCALE_LEVELS] = {0.340121,0.384804,0.280073,0.347667,0.372019,0.419144,0.335678,0.336661,0.313423,0.377457,0.438418,0.371464,0.387988};
        //float costs_classify    [NUM_SCALE_LEVELS] = {0.777011,0.814835,1.345708,0.88426,1.400191,0.903919,0.925312,0.90149,0.829934,0.723972,0.742641,0.720691,0.905177};
        //float cost_collect_locations               = 3.967353;

        /* | first graph release      | second graph release     | first graph release again
         *  <---------PERIOD--------->
         *  <--------------- PERIOD * args.num_fine_graphs ---------->
         */
        int period = PERIOD * args.num_fine_graphs;
        int m_cpus = 16;
        struct task_info t_info;
        t_info.early = args.early;
        t_info.realtime = args.realtime;
        t_info.sched = fine_CDE;
        t_info.period = period;
        t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, cost_color_convert);
        t_info.phase = PERIOD * g_idx;
        t_info.id = 0;
        if (args.cluster != -1)
            t_info.cluster = args.cluster;
        else
            t_info.cluster = args.cluster;
        *t0 = new thread(&App::thread_color_convert, this,
                &color_convert_node, fine_init_barrier,
                gpu_hog, cpu_hog, frames, t_info, g_idx);

        t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, cost_compute_scales);
        t_info.id = 1;
        t_info.phase = t_info.phase + bound_color_convert;
        *t1 = new thread(&cv::cuda::HOG::thread_fine_compute_scales, gpu_hog,
                &compute_scales_node, fine_init_barrier, t_info);

        for (int i=0; i<NUM_SCALE_LEVELS; i++) {
            t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, costs_resize[i]);
            t_info.id = 0 * NUM_SCALE_LEVELS + i + 2;
            t_info.phase = PERIOD * g_idx + bound_color_convert + bound_compute_scales;
            t_info.s_info_in = in_sync_info_resize + i;
            t_info.s_info_out = out_sync_info_resize + i;
            t2[i] = new thread(&cv::cuda::HOG::thread_fine_resize, gpu_hog,
                    resize_node + i, fine_init_barrier, t_info);

            t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, costs_compute_grad[i]);
            t_info.id = 1 * NUM_SCALE_LEVELS + i + 2;
            t_info.phase = t_info.phase + bounds_resize[i];
            t_info.s_info_in = in_sync_info_compute_gradients + i;
            t_info.s_info_out = out_sync_info_compute_gradients + i;
            t3[i] = new thread(&cv::cuda::HOG::thread_fine_compute_gradients,
                    gpu_hog, compute_gradients_node + i, fine_init_barrier, t_info);

            t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, costs_CDE[i]);
            t_info.id = 2 * NUM_SCALE_LEVELS + i + 2;
            t_info.phase = t_info.phase + bounds_compute_grad[i];
            t_info.s_info_in = in_sync_info_CDE + i;
            t_info.s_info_out = out_sync_info_CDE + i;
            t456[i] = new thread(&cv::cuda::HOG::thread_fine_CDE,
                    gpu_hog, CDE_node + i, fine_init_barrier, t_info);
        }

        t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, cost_collect_locations);
        t_info.id = 3 * NUM_SCALE_LEVELS + 2;
        t_info.phase = t_info.phase + *std::max_element(bounds_CDE, bounds_CDE+NUM_SCALE_LEVELS);
        *t7 = new thread(&cv::cuda::HOG::thread_fine_collect_locations, gpu_hog,
                &collect_locations_node, fine_init_barrier, t_info);
        //t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, cost_display);
        //t_info.phase;
        *t8 = new thread(&App::thread_display, this, &display_node,
                fine_init_barrier, g_idx == 0 && args.display);
    }

    /* graph construction finishes */

    printf("Joining pthreads...\n");

    for (int g_idx = 0; g_idx < args.num_fine_graphs; g_idx++) {
        graph_t g = arr_g[g_idx];
        thread* t0 = arr_t0[g_idx];
        thread* t1 = arr_t1[g_idx];
        thread** t2 = arr_t2 + g_idx * NUM_SCALE_LEVELS;
        thread** t3 = arr_t3 + g_idx * NUM_SCALE_LEVELS;
        thread** t456 = arr_t456 + g_idx * NUM_SCALE_LEVELS;
        thread* t7 = arr_t7[g_idx];
        thread* t8 = arr_t8[g_idx];
        t0->join();
        t1->join();
        delete t0;
        delete t1;
        for (int i=0; i<NUM_SCALE_LEVELS; i++) {
            if (t2[i]->joinable()) t2[i]->join();
            if (t3[i]->joinable()) t3[i]->join();
            if (t456[i]->joinable()) t456[i]->join();

            delete t2[i];
            delete t3[i];
            delete t456[i];
        }
        t7->join();
        t8->join();
        delete t7;
        delete t8;
        CheckError(pgm_destroy_graph(g));
    }
    free(arr_t1);
    free(arr_t2);
    free(arr_t3);
    free(arr_t456);
    free(arr_t7);
    free(arr_t8);

    //CheckError(pgm_destroy_graph(g));
    CheckError(pgm_destroy());
    fprintf(stdout, "cleaned up ...");
}

void App::sched_ABCD_hog(cv::Ptr<cv::cuda::HOG> gpu_hog, cv::HOGDescriptor cpu_hog, Mat* frames)
{
    pthread_barrier_t arr_fine_init_barrier[args.num_fine_graphs];
    /* graph construction */
    graph_t arr_g [args.num_fine_graphs];

    struct sync_info arr_sync_info_ABCD     [args.num_fine_graphs][NUM_SCALE_LEVELS];
    struct sync_info arr_sync_info_classify [args.num_fine_graphs][NUM_SCALE_LEVELS];

    thread** arr_t0    = (thread**) calloc(args.num_fine_graphs, sizeof(std::thread *));
    thread** arr_t1    = (thread**) calloc(args.num_fine_graphs, sizeof(std::thread *));
    thread** arr_t2345 = (thread**) calloc(args.num_fine_graphs * NUM_SCALE_LEVELS, sizeof(std::thread *));
    thread** arr_t6    = (thread**) calloc(args.num_fine_graphs * NUM_SCALE_LEVELS, sizeof(std::thread *));
    thread** arr_t7    = (thread**) calloc(args.num_fine_graphs, sizeof(std::thread *));
    thread** arr_t8    = (thread**) calloc(args.num_fine_graphs, sizeof(std::thread *));

    char buf[30];
    sprintf(buf, "/tmp/graph_t%d", args.task_id);
    CheckError(pgm_init(buf, 1));

    for (int g_idx = 0; g_idx < args.num_fine_graphs; g_idx++)
    {
        pthread_barrier_t* fine_init_barrier = arr_fine_init_barrier + g_idx;

        // A graph consists of the graph itself, nodes, and edge
        graph_t* g_ptr = arr_g + g_idx;

        node_t color_convert_node;
        node_t compute_scales_node;
        node_t ABCD_node               [NUM_SCALE_LEVELS]; // resize -> normalize-hists
        node_t classify_node           [NUM_SCALE_LEVELS];
        node_t collect_locations_node;
        node_t display_node;

        edge_t e0_1;
        edge_t e1_2345 [NUM_SCALE_LEVELS];
        edge_t e2345_6 [NUM_SCALE_LEVELS];
        edge_t e6_7    [NUM_SCALE_LEVELS];
        edge_t e7_8;

        // Initialize the graph
        sprintf(buf, "hog_%d", g_idx);
        CheckError(pgm_init_graph(g_ptr, buf));
        graph_t g = *g_ptr;

        // Initialize the nodes
        CheckError(pgm_init_node(&color_convert_node, g, "color_convert"));
        CheckError(pgm_init_node(&compute_scales_node, g, "compute_scales"));
        for (int i=0; i<NUM_SCALE_LEVELS; i++) {
            CheckError(pgm_init_node(ABCD_node + i, g, "ABCD"));
            CheckError(pgm_init_node(classify_node + i, g, "classify"));
        }
        CheckError(pgm_init_node(&collect_locations_node, g, "collect_locations"));
        CheckError(pgm_init_node(&display_node, g, "display"));

        // Initialize the edges
        edge_attr_t fast_mq_attr;
        memset(&fast_mq_attr, 0, sizeof(fast_mq_attr));
        fast_mq_attr.type = pgm_fast_fifo_edge;

        fast_mq_attr.nr_produce = sizeof(struct params_compute);
        fast_mq_attr.nr_consume = sizeof(struct params_compute);
        fast_mq_attr.nr_threshold = sizeof(struct params_compute);
        CheckError(pgm_init_edge(&e0_1, color_convert_node, compute_scales_node, "e0_1", &fast_mq_attr));

        for (int i=0; i<NUM_SCALE_LEVELS; i++) {
            sprintf(buf, "e1_2345_%d", i);
            fast_mq_attr.nr_produce = sizeof(struct params_resize);
            fast_mq_attr.nr_consume = sizeof(struct params_resize);
            fast_mq_attr.nr_threshold = sizeof(struct params_resize);
            CheckError(pgm_init_edge(e1_2345 + i,
                        compute_scales_node,
                        ABCD_node[i], buf, &fast_mq_attr));

            sprintf(buf, "e2345_6_%d", i);
            fast_mq_attr.nr_produce = sizeof(struct params_fine_classify);
            fast_mq_attr.nr_consume = sizeof(struct params_fine_classify);
            fast_mq_attr.nr_threshold = sizeof(struct params_fine_classify);
            CheckError(pgm_init_edge(e2345_6 + i,
                        ABCD_node[i],
                        classify_node[i], buf, &fast_mq_attr));

            sprintf(buf, "e6_7_%d", i);
            fast_mq_attr.nr_produce = sizeof(struct params_fine_collect_locations);
            fast_mq_attr.nr_consume = sizeof(struct params_fine_collect_locations);
            fast_mq_attr.nr_threshold = sizeof(struct params_fine_collect_locations);
            CheckError(pgm_init_edge(e6_7 + i,
                        classify_node[i],
                        collect_locations_node, buf, &fast_mq_attr));
        }

        fast_mq_attr.nr_produce = sizeof(struct params_display);
        fast_mq_attr.nr_consume = sizeof(struct params_display);
        fast_mq_attr.nr_threshold = sizeof(struct params_display);
        CheckError(pgm_init_edge(&e7_8, collect_locations_node, display_node, "e7_8", &fast_mq_attr));

        pthread_barrier_init(fine_init_barrier, 0, 2 * NUM_SCALE_LEVELS + 4);

        thread** t0  = arr_t0 + g_idx;
        thread** t1  = arr_t1 + g_idx;
        thread** t2345 = arr_t2345 + g_idx * NUM_SCALE_LEVELS;
        thread** t6  = arr_t6 + g_idx * NUM_SCALE_LEVELS;
        thread** t7  = arr_t7 + g_idx;
        thread** t8  = arr_t8 + g_idx;

        struct sync_info* in_sync_info_ABCD               = arr_sync_info_ABCD              [((g_idx + args.num_fine_graphs - 1) % args.num_fine_graphs)];
        struct sync_info* in_sync_info_classify           = arr_sync_info_classify          [((g_idx + args.num_fine_graphs - 1) % args.num_fine_graphs)];

        struct sync_info* out_sync_info_ABCD               = arr_sync_info_ABCD              [g_idx];
        struct sync_info* out_sync_info_classify           = arr_sync_info_classify          [g_idx];

        for (int i=0; i<NUM_SCALE_LEVELS; i++) {
            sync_info_init(i + in_sync_info_ABCD);
            sync_info_init(i + in_sync_info_classify);

            sync_info_init(i + out_sync_info_ABCD);
            sync_info_init(i + out_sync_info_classify);
        }

        /* G-FL fine-grained HOG */
        float bound_color_convert                   = 31.0194960215;
        float bound_compute_scales                  = 30.5688293615;
        // float bounds_resize      [NUM_SCALE_LEVELS] = {30.2419972231,30.3148010256,30.2936775569,30.3079650324,30.2963991483,30.3058228698,30.3005336032,30.2888197835,30.293491626,30.2840584057,30.3013304497,30.3016323015,30.3111116087};
        // float bounds_compute_grad[NUM_SCALE_LEVELS] = {30.4902863291,30.44552396,30.4706995604,30.4291495557,30.4171804989,30.4135009326,30.3785928987,30.4059076714,30.3816881109,30.3627509567,30.3564101342,30.3519205285,30.3405616838};
        // float bounds_compute_hist[NUM_SCALE_LEVELS] = {30.5326861275,30.4951636278,30.5415047381,30.4958614402,30.5149987735,30.4750125581,30.5373623676,30.522424222,30.4851937235,30.4577396346,30.4263855713,30.4118414514,30.5202006157};
        float bounds_ABCD        [NUM_SCALE_LEVELS] = {30.2419972231,30.3148010256,30.2936775569,30.3079650324,30.2963991483,30.3058228698,30.3005336032,30.2888197835,30.293491626,30.2840584057,30.3013304497,30.3016323015,30.3111116087};
        // float bounds_normalize   [NUM_SCALE_LEVELS] = {30.2903383654,30.3080869341,30.2928467608,30.3064130287,30.3114078316,30.3120987837,30.2985990076,30.2803770804,30.3099685825,30.3123524377,30.2989110618,30.2943410679,30.3081075149};
        float bounds_classify    [NUM_SCALE_LEVELS] = {30.4792191337,30.4948973087,30.4540269983,30.429525815,30.5229441951,30.5070612675,30.497249677,30.4955912512,30.4267037822,30.4508744413,30.430038752,30.3991297266,30.3767439685};
        //float bound_collect_locations               30.983976896;

        /* G-FL fine-grained HOG */
        float cost_color_convert                   = 4.509346;
        float cost_compute_scales                  = 1.947347;
        // float costs_resize      [NUM_SCALE_LEVELS] = {0.089336,0.503219,0.383134,0.464357,0.398606,0.452179,0.42211,0.355518,0.382077,0.32845,0.42664,0.428356,0.482245};
        // float costs_compute_grad[NUM_SCALE_LEVELS] = {1.500837,1.246367,1.389488,1.15328,1.085237,1.064319,0.86587,1.021152,0.883466,0.77581,0.739763,0.71424,0.649666};
        // float costs_compute_hist[NUM_SCALE_LEVELS] = {1.741876,1.528564,1.792009,1.532531,1.641325,1.414007,1.76846,1.683538,1.471886,1.315812,1.137567,1.054885,1.670897};
        float costs_ABCD        [NUM_SCALE_LEVELS] = {0.089336,0.503219,0.383134,0.464357,0.398606,0.452179,0.42211,0.355518,0.382077,0.32845,0.42664,0.428356,0.482245};
        // float costs_normalize   [NUM_SCALE_LEVELS] = {0.364151,0.46505,0.378411,0.455534,0.483929,0.487857,0.411112,0.307522,0.475747,0.489299,0.412886,0.386906,0.465167};
        float costs_classify    [NUM_SCALE_LEVELS] = {1.437921,1.52705,1.294706,1.155419,1.686494,1.596201,1.540423,1.530995,1.139376,1.276784,1.158335,0.98262,0.855359};
        float cost_collect_locations               = 4.307423;

        /* C-FL fine-grained HOG */
        //float bound_color_convert                   = 26.3496352098;
        //float bound_compute_scales                  = 25.6841982494;
        //float bounds_resize      [NUM_SCALE_LEVELS] = {25.0939476981,25.1387935171,25.130189875,25.1357033275,25.1312621102,25.1463332249,25.1316714257,25.1302245094,25.1267072576,25.1262401226,25.1348909935,25.1432006726,25.1413123823};
        //float bounds_compute_grad[NUM_SCALE_LEVELS] = {25.3489965402,25.4185896374,25.4092134479,25.4079932294,25.4051932818,25.4030496702,25.312479578,25.291207475,25.2868561354,25.2685316772,25.2518075571,25.246440943,25.2364255915};
        //float bounds_compute_hist[NUM_SCALE_LEVELS] = {25.5728014354,25.4973631396,25.5123077387,25.471628638,25.3841716342,25.3274278981,25.4720253593,25.5328196023,25.4654399579,25.3426074958,25.4860697502,25.3209827517,25.4610373822};
        //float bounds_ABCD        [NUM_SCALE_LEVELS] = {25.0939476981,25.1387935171,25.130189875,25.1357033275,25.1312621102,25.1463332249,25.1316714257,25.1302245094,25.1267072576,25.1262401226,25.1348909935,25.1432006726,25.1413123823};
        //float bounds_normalize   [NUM_SCALE_LEVELS] = {25.1658996554,25.1786894798,25.1487118351,25.1680595823,25.1750299693,25.1885187788,25.1646279147,25.1649092834,25.1582577618,25.1765865135,25.1940356661,25.1748711091,25.179600851};
        //float bounds_classify    [NUM_SCALE_LEVELS] = {25.2909527262,25.3017792668,25.4537335318,25.3216511097,25.4693284561,25.3272781973,25.3334016159,25.3265829333,25.3061011243,25.2757711248,25.2811148401,25.2748319888,25.3276382806};
        ////float bound_collect_locations               26.204139258;

        /////* C-FL fine-grained HOG */
        //float cost_color_convert                   = 4.475663;
        //float cost_compute_scales                  = 2.150868;
        //float costs_resize      [NUM_SCALE_LEVELS] = {0.088747,0.245422,0.215364,0.234626,0.21911,0.271763,0.22054,0.215485,0.203197,0.201565,0.231788,0.260819,0.254222};
        //float costs_compute_grad[NUM_SCALE_LEVELS] = {0.979795,1.222928,1.190171,1.185908,1.176126,1.168637,0.852218,0.777901,0.762699,0.69868,0.640252,0.621503,0.586513};
        //float costs_compute_hist[NUM_SCALE_LEVELS] = {1.761688,1.498134,1.550345,1.408227,1.102684,0.904442,1.409613,1.622006,1.386606,0.957474,1.458679,0.881925,1.371225};
        //float costs_ABCD        [NUM_SCALE_LEVELS] = {0.088747,0.245422,0.215364,0.234626,0.21911,0.271763,0.22054,0.215485,0.203197,0.201565,0.231788,0.260819,0.254222};
        //float costs_normalize   [NUM_SCALE_LEVELS] = {0.340121,0.384804,0.280073,0.347667,0.372019,0.419144,0.335678,0.336661,0.313423,0.377457,0.438418,0.371464,0.387988};
        //float costs_classify    [NUM_SCALE_LEVELS] = {0.777011,0.814835,1.345708,0.88426,1.400191,0.903919,0.925312,0.90149,0.829934,0.723972,0.742641,0.720691,0.905177};
        //float cost_collect_locations               = 3.967353;

        /* | first graph release      | second graph release     | first graph release again
         *  <---------PERIOD--------->
         *  <--------------- PERIOD * args.num_fine_graphs ---------->
         */
        int period = PERIOD * args.num_fine_graphs;
        int m_cpus = 16;
        struct task_info t_info;
        t_info.early = args.early;
        t_info.realtime = args.realtime;
        t_info.sched = fine_ABCD;
        t_info.period = period;
        t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, cost_color_convert);
        t_info.phase = PERIOD * g_idx;
        t_info.id = 0;
        if (args.cluster != -1)
            t_info.cluster = args.cluster;
        else
            t_info.cluster = args.cluster;
        *t0 = new thread(&App::thread_color_convert, this,
                &color_convert_node, fine_init_barrier,
                gpu_hog, cpu_hog, frames, t_info, g_idx);

        t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, cost_compute_scales);
        t_info.id = 1;
        t_info.phase = t_info.phase + bound_color_convert;
        *t1 = new thread(&cv::cuda::HOG::thread_fine_compute_scales, gpu_hog,
                &compute_scales_node, fine_init_barrier, t_info);

        for (int i=0; i<NUM_SCALE_LEVELS; i++) {
            t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, costs_ABCD[i]);
            t_info.id = 0 * NUM_SCALE_LEVELS + i + 2;
            t_info.phase = PERIOD * g_idx + bound_color_convert + bound_compute_scales;
            t_info.s_info_in = in_sync_info_ABCD + i;
            t_info.s_info_out = out_sync_info_ABCD + i;
            t2345[i] = new thread(&cv::cuda::HOG::thread_fine_ABCD, gpu_hog,
                    ABCD_node + i, fine_init_barrier, t_info);

            t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, costs_classify[i]);
            t_info.id = 1 * NUM_SCALE_LEVELS + i + 2;
            t_info.phase = t_info.phase + bounds_ABCD[i];
            t_info.s_info_in = in_sync_info_classify + i;
            t_info.s_info_out = out_sync_info_classify + i;
            t6[i] = new thread(&cv::cuda::HOG::thread_fine_classify, gpu_hog,
                    classify_node + i, fine_init_barrier, t_info);
        }

        t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, cost_collect_locations);
        t_info.id = 2 * NUM_SCALE_LEVELS + 2;
        t_info.phase = t_info.phase + *std::max_element(bounds_classify, bounds_classify+NUM_SCALE_LEVELS);
        *t7 = new thread(&cv::cuda::HOG::thread_fine_collect_locations, gpu_hog,
                &collect_locations_node, fine_init_barrier, t_info);
        //t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, cost_display);
        //t_info.phase;
        *t8 = new thread(&App::thread_display, this, &display_node,
                fine_init_barrier, g_idx == 0 && args.display);
    }

    /* graph construction finishes */

    printf("Joining pthreads...\n");

    for (int g_idx = 0; g_idx < args.num_fine_graphs; g_idx++) {
        graph_t g = arr_g[g_idx];
        thread* t0 = arr_t0[g_idx];
        thread* t1 = arr_t1[g_idx];
        thread** t2345 = arr_t2345 + g_idx * NUM_SCALE_LEVELS;
        thread** t6  = arr_t6 + g_idx * NUM_SCALE_LEVELS;
        thread* t7 = arr_t7[g_idx];
        thread* t8 = arr_t8[g_idx];
        t0->join();
        t1->join();
        delete t0;
        delete t1;
        for (int i=0; i<NUM_SCALE_LEVELS; i++) {
            if (t2345[i]->joinable()) t2345[i]->join();
            if (t6[i]->joinable()) t6[i]->join();

            delete t2345[i];
            delete t6[i];
        }
        t7->join();
        t8->join();
        delete t7;
        delete t8;
        CheckError(pgm_destroy_graph(g));
    }
    free(arr_t1);
    free(arr_t2345);
    free(arr_t6);
    free(arr_t7);
    free(arr_t8);

    //CheckError(pgm_destroy_graph(g));
    CheckError(pgm_destroy());
    fprintf(stdout, "cleaned up ...");
}

void App::sched_BCDE_hog(cv::Ptr<cv::cuda::HOG> gpu_hog, cv::HOGDescriptor cpu_hog, Mat* frames)
{
    pthread_barrier_t arr_fine_init_barrier[args.num_fine_graphs];
    /* graph construction */
    graph_t arr_g [args.num_fine_graphs];

    struct sync_info arr_sync_info_resize [args.num_fine_graphs][NUM_SCALE_LEVELS];
    struct sync_info arr_sync_info_BCDE   [args.num_fine_graphs][NUM_SCALE_LEVELS];

    thread** arr_t0    = (thread**) calloc(args.num_fine_graphs, sizeof(std::thread *));
    thread** arr_t1    = (thread**) calloc(args.num_fine_graphs, sizeof(std::thread *));
    thread** arr_t2    = (thread**) calloc(args.num_fine_graphs * NUM_SCALE_LEVELS, sizeof(std::thread *));
    thread** arr_t3456 = (thread**) calloc(args.num_fine_graphs * NUM_SCALE_LEVELS, sizeof(std::thread *));
    thread** arr_t7    = (thread**) calloc(args.num_fine_graphs, sizeof(std::thread *));
    thread** arr_t8    = (thread**) calloc(args.num_fine_graphs, sizeof(std::thread *));

    char buf[30];
    sprintf(buf, "/tmp/graph_t%d", args.task_id);
    CheckError(pgm_init(buf, 1));

    for (int g_idx = 0; g_idx < args.num_fine_graphs; g_idx++)
    {
        pthread_barrier_t* fine_init_barrier = arr_fine_init_barrier + g_idx;

        // A graph consists of the graph itself, nodes, and edge
        graph_t* g_ptr = arr_g + g_idx;

        node_t color_convert_node;
        node_t compute_scales_node;
        node_t resize_node             [NUM_SCALE_LEVELS];
        node_t BCDE_node               [NUM_SCALE_LEVELS]; // compute-grads -> classify-hists
        node_t collect_locations_node;
        node_t display_node;

        edge_t e0_1;
        edge_t e1_2    [NUM_SCALE_LEVELS];
        edge_t e2_3456 [NUM_SCALE_LEVELS];
        edge_t e3456_7 [NUM_SCALE_LEVELS];
        edge_t e7_8;

        // Initialize the graph
        sprintf(buf, "hog_%d", g_idx);
        CheckError(pgm_init_graph(g_ptr, buf));
        graph_t g = *g_ptr;

        // Initialize the nodes
        CheckError(pgm_init_node(&color_convert_node, g, "color_convert"));
        CheckError(pgm_init_node(&compute_scales_node, g, "compute_scales"));
        for (int i=0; i<NUM_SCALE_LEVELS; i++) {
            CheckError(pgm_init_node(resize_node + i, g, "resize"));
            CheckError(pgm_init_node(BCDE_node + i, g, "BCD"));
        }
        CheckError(pgm_init_node(&collect_locations_node, g, "collect_locations"));
        CheckError(pgm_init_node(&display_node, g, "display"));

        // Initialize the edges
        edge_attr_t fast_mq_attr;
        memset(&fast_mq_attr, 0, sizeof(fast_mq_attr));
        fast_mq_attr.type = pgm_fast_fifo_edge;

        fast_mq_attr.nr_produce = sizeof(struct params_compute);
        fast_mq_attr.nr_consume = sizeof(struct params_compute);
        fast_mq_attr.nr_threshold = sizeof(struct params_compute);
        CheckError(pgm_init_edge(&e0_1, color_convert_node, compute_scales_node, "e0_1", &fast_mq_attr));

        for (int i=0; i<NUM_SCALE_LEVELS; i++) {
            sprintf(buf, "e1_2_%d", i);
            fast_mq_attr.nr_produce = sizeof(struct params_resize);
            fast_mq_attr.nr_consume = sizeof(struct params_resize);
            fast_mq_attr.nr_threshold = sizeof(struct params_resize);
            CheckError(pgm_init_edge(e1_2 + i,
                        compute_scales_node,
                        resize_node[i], buf, &fast_mq_attr));

            sprintf(buf, "e2_3456_%d", i);
            fast_mq_attr.nr_produce = sizeof(struct params_compute_gradients);
            fast_mq_attr.nr_consume = sizeof(struct params_compute_gradients);
            fast_mq_attr.nr_threshold = sizeof(struct params_compute_gradients);
            CheckError(pgm_init_edge(e2_3456 + i,
                        resize_node[i],
                        BCDE_node[i], buf, &fast_mq_attr));

            sprintf(buf, "e3456_7_%d", i);
            fast_mq_attr.nr_produce = sizeof(struct params_fine_collect_locations);
            fast_mq_attr.nr_consume = sizeof(struct params_fine_collect_locations);
            fast_mq_attr.nr_threshold = sizeof(struct params_fine_collect_locations);
            CheckError(pgm_init_edge(e3456_7 + i,
                        BCDE_node[i],
                        collect_locations_node, buf, &fast_mq_attr));
        }

        fast_mq_attr.nr_produce = sizeof(struct params_display);
        fast_mq_attr.nr_consume = sizeof(struct params_display);
        fast_mq_attr.nr_threshold = sizeof(struct params_display);
        CheckError(pgm_init_edge(&e7_8, collect_locations_node, display_node, "e7_8", &fast_mq_attr));

        pthread_barrier_init(fine_init_barrier, 0, 2 * NUM_SCALE_LEVELS + 4);

        thread** t0    = arr_t0 + g_idx;
        thread** t1    = arr_t1 + g_idx;
        thread** t2    = arr_t2 + g_idx * NUM_SCALE_LEVELS;
        thread** t3456 = arr_t3456 + g_idx * NUM_SCALE_LEVELS;
        thread** t7    = arr_t7 + g_idx;
        thread** t8    = arr_t8 + g_idx;

        struct sync_info* in_sync_info_resize             = arr_sync_info_resize            [((g_idx + args.num_fine_graphs - 1) % args.num_fine_graphs)];
        struct sync_info* in_sync_info_BCDE               = arr_sync_info_BCDE              [((g_idx + args.num_fine_graphs - 1) % args.num_fine_graphs)];

        struct sync_info* out_sync_info_resize             = arr_sync_info_resize            [g_idx];
        struct sync_info* out_sync_info_BCDE               = arr_sync_info_BCDE              [g_idx];

        for (int i=0; i<NUM_SCALE_LEVELS; i++) {
            sync_info_init(i + in_sync_info_resize);
            sync_info_init(i + in_sync_info_BCDE);

            sync_info_init(i + out_sync_info_resize);
            sync_info_init(i + out_sync_info_BCDE);
        }

        /* G-FL fine-grained HOG */
        float bound_color_convert                   = 31.0194960215;
        float bound_compute_scales                  = 30.5688293615;
        float bounds_resize      [NUM_SCALE_LEVELS] = {30.2419972231,30.3148010256,30.2936775569,30.3079650324,30.2963991483,30.3058228698,30.3005336032,30.2888197835,30.293491626,30.2840584057,30.3013304497,30.3016323015,30.3111116087};
        // float bounds_compute_grad[NUM_SCALE_LEVELS] = {30.4902863291,30.44552396,30.4706995604,30.4291495557,30.4171804989,30.4135009326,30.3785928987,30.4059076714,30.3816881109,30.3627509567,30.3564101342,30.3519205285,30.3405616838};
        // float bounds_compute_hist[NUM_SCALE_LEVELS] = {30.5326861275,30.4951636278,30.5415047381,30.4958614402,30.5149987735,30.4750125581,30.5373623676,30.522424222,30.4851937235,30.4577396346,30.4263855713,30.4118414514,30.5202006157};
        float bounds_BCDE        [NUM_SCALE_LEVELS] = {30.4902863291,30.44552396,30.4706995604,30.4291495557,30.4171804989,30.4135009326,30.3785928987,30.4059076714,30.3816881109,30.3627509567,30.3564101342,30.3519205285,30.3405616838};
        // float bounds_normalize   [NUM_SCALE_LEVELS] = {30.2903383654,30.3080869341,30.2928467608,30.3064130287,30.3114078316,30.3120987837,30.2985990076,30.2803770804,30.3099685825,30.3123524377,30.2989110618,30.2943410679,30.3081075149};
        // float bounds_classify    [NUM_SCALE_LEVELS] = {30.4792191337,30.4948973087,30.4540269983,30.429525815,30.5229441951,30.5070612675,30.497249677,30.4955912512,30.4267037822,30.4508744413,30.430038752,30.3991297266,30.3767439685};
        //float bound_collect_locations               30.983976896;

        /* G-FL fine-grained HOG */
        float cost_color_convert                   = 4.509346;
        float cost_compute_scales                  = 1.947347;
        float costs_resize      [NUM_SCALE_LEVELS] = {0.089336,0.503219,0.383134,0.464357,0.398606,0.452179,0.42211,0.355518,0.382077,0.32845,0.42664,0.428356,0.482245};
        // float costs_compute_grad[NUM_SCALE_LEVELS] = {1.500837,1.246367,1.389488,1.15328,1.085237,1.064319,0.86587,1.021152,0.883466,0.77581,0.739763,0.71424,0.649666};
        // float costs_compute_hist[NUM_SCALE_LEVELS] = {1.741876,1.528564,1.792009,1.532531,1.641325,1.414007,1.76846,1.683538,1.471886,1.315812,1.137567,1.054885,1.670897};
        float costs_BCDE        [NUM_SCALE_LEVELS] = {1.500837,1.246367,1.389488,1.15328,1.085237,1.064319,0.86587,1.021152,0.883466,0.77581,0.739763,0.71424,0.649666};
        // float costs_normalize   [NUM_SCALE_LEVELS] = {0.364151,0.46505,0.378411,0.455534,0.483929,0.487857,0.411112,0.307522,0.475747,0.489299,0.412886,0.386906,0.465167};
        // float costs_classify    [NUM_SCALE_LEVELS] = {1.437921,1.52705,1.294706,1.155419,1.686494,1.596201,1.540423,1.530995,1.139376,1.276784,1.158335,0.98262,0.855359};
        float cost_collect_locations               = 4.307423;

        /* C-FL fine-grained HOG */
        //float bound_color_convert                   = 26.3496352098;
        //float bound_compute_scales                  = 25.6841982494;
        //float bounds_resize      [NUM_SCALE_LEVELS] = {25.0939476981,25.1387935171,25.130189875,25.1357033275,25.1312621102,25.1463332249,25.1316714257,25.1302245094,25.1267072576,25.1262401226,25.1348909935,25.1432006726,25.1413123823};
        //float bounds_compute_grad[NUM_SCALE_LEVELS] = {25.3489965402,25.4185896374,25.4092134479,25.4079932294,25.4051932818,25.4030496702,25.312479578,25.291207475,25.2868561354,25.2685316772,25.2518075571,25.246440943,25.2364255915};
        //float bounds_BCDE        [NUM_SCALE_LEVELS] = {25.3489965402,25.4185896374,25.4092134479,25.4079932294,25.4051932818,25.4030496702,25.312479578,25.291207475,25.2868561354,25.2685316772,25.2518075571,25.246440943,25.2364255915};
        //float bounds_compute_hist[NUM_SCALE_LEVELS] = {25.5728014354,25.4973631396,25.5123077387,25.471628638,25.3841716342,25.3274278981,25.4720253593,25.5328196023,25.4654399579,25.3426074958,25.4860697502,25.3209827517,25.4610373822};
        //float bounds_normalize   [NUM_SCALE_LEVELS] = {25.1658996554,25.1786894798,25.1487118351,25.1680595823,25.1750299693,25.1885187788,25.1646279147,25.1649092834,25.1582577618,25.1765865135,25.1940356661,25.1748711091,25.179600851};
        //float bounds_classify    [NUM_SCALE_LEVELS] = {25.2909527262,25.3017792668,25.4537335318,25.3216511097,25.4693284561,25.3272781973,25.3334016159,25.3265829333,25.3061011243,25.2757711248,25.2811148401,25.2748319888,25.3276382806};
        ////float bound_collect_locations               26.204139258;

        /////* C-FL fine-grained HOG */
        //float cost_color_convert                   = 4.475663;
        //float cost_compute_scales                  = 2.150868;
        //float costs_resize      [NUM_SCALE_LEVELS] = {0.088747,0.245422,0.215364,0.234626,0.21911,0.271763,0.22054,0.215485,0.203197,0.201565,0.231788,0.260819,0.254222};
        //float costs_compute_grad[NUM_SCALE_LEVELS] = {0.979795,1.222928,1.190171,1.185908,1.176126,1.168637,0.852218,0.777901,0.762699,0.69868,0.640252,0.621503,0.586513};
        //float costs_BCDE        [NUM_SCALE_LEVELS] = {0.979795,1.222928,1.190171,1.185908,1.176126,1.168637,0.852218,0.777901,0.762699,0.69868,0.640252,0.621503,0.586513};
        //float costs_compute_hist[NUM_SCALE_LEVELS] = {1.761688,1.498134,1.550345,1.408227,1.102684,0.904442,1.409613,1.622006,1.386606,0.957474,1.458679,0.881925,1.371225};
        //float costs_normalize   [NUM_SCALE_LEVELS] = {0.340121,0.384804,0.280073,0.347667,0.372019,0.419144,0.335678,0.336661,0.313423,0.377457,0.438418,0.371464,0.387988};
        //float costs_classify    [NUM_SCALE_LEVELS] = {0.777011,0.814835,1.345708,0.88426,1.400191,0.903919,0.925312,0.90149,0.829934,0.723972,0.742641,0.720691,0.905177};
        //float cost_collect_locations               = 3.967353;

        /* | first graph release      | second graph release     | first graph release again
         *  <---------PERIOD--------->
         *  <--------------- PERIOD * args.num_fine_graphs ---------->
         */
        int period = PERIOD * args.num_fine_graphs;
        int m_cpus = 16;
        struct task_info t_info;
        t_info.early = args.early;
        t_info.realtime = args.realtime;
        t_info.sched = fine_BCDE;
        t_info.period = period;
        t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, cost_color_convert);
        t_info.phase = PERIOD * g_idx;
        t_info.id = 0;
        if (args.cluster != -1)
            t_info.cluster = args.cluster;
        else
            t_info.cluster = args.cluster;
        *t0 = new thread(&App::thread_color_convert, this,
                &color_convert_node, fine_init_barrier,
                gpu_hog, cpu_hog, frames, t_info, g_idx);

        t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, cost_compute_scales);
        t_info.id = 1;
        t_info.phase = t_info.phase + bound_color_convert;
        *t1 = new thread(&cv::cuda::HOG::thread_fine_compute_scales, gpu_hog,
                &compute_scales_node, fine_init_barrier, t_info);

        for (int i=0; i<NUM_SCALE_LEVELS; i++) {
            t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, costs_resize[i]);
            t_info.id = 0 * NUM_SCALE_LEVELS + i + 2;
            t_info.phase = PERIOD * g_idx + bound_color_convert + bound_compute_scales;
            t_info.s_info_in = in_sync_info_resize + i;
            t_info.s_info_out = out_sync_info_resize + i;
            t2[i] = new thread(&cv::cuda::HOG::thread_fine_resize, gpu_hog,
                    resize_node + i, fine_init_barrier, t_info);

            t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, costs_BCDE[i]);
            t_info.id = 1 * NUM_SCALE_LEVELS + i + 2;
            t_info.phase = PERIOD * g_idx + bounds_resize[i];
            t_info.s_info_in = in_sync_info_BCDE + i;
            t_info.s_info_out = out_sync_info_BCDE + i;
            t3456[i] = new thread(&cv::cuda::HOG::thread_fine_BCDE, gpu_hog,
                    BCDE_node + i, fine_init_barrier, t_info);
        }

        t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, cost_collect_locations);
        t_info.id = 2 * NUM_SCALE_LEVELS + 2;
        t_info.phase = t_info.phase + *std::max_element(bounds_BCDE, bounds_BCDE+NUM_SCALE_LEVELS);
        *t7 = new thread(&cv::cuda::HOG::thread_fine_collect_locations, gpu_hog,
                &collect_locations_node, fine_init_barrier, t_info);
        //t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, cost_display);
        //t_info.phase;
        *t8 = new thread(&App::thread_display, this, &display_node,
                fine_init_barrier, g_idx == 0 && args.display);
    }

    /* graph construction finishes */

    printf("Joining pthreads...\n");

    for (int g_idx = 0; g_idx < args.num_fine_graphs; g_idx++) {
        graph_t g = arr_g[g_idx];
        thread* t0 = arr_t0[g_idx];
        thread* t1 = arr_t1[g_idx];
        thread** t2 = arr_t2 + g_idx * NUM_SCALE_LEVELS;
        thread** t3456 = arr_t3456 + g_idx * NUM_SCALE_LEVELS;
        thread* t7 = arr_t7[g_idx];
        thread* t8 = arr_t8[g_idx];
        t0->join();
        t1->join();
        delete t0;
        delete t1;
        for (int i=0; i<NUM_SCALE_LEVELS; i++) {
            if (t2[i]->joinable()) t2[i]->join();
            if (t3456[i]->joinable()) t3456[i]->join();

            delete t2[i];
            delete t3456[i];
        }
        t7->join();
        t8->join();
        delete t7;
        delete t8;
        CheckError(pgm_destroy_graph(g));
    }
    free(arr_t1);
    free(arr_t2);
    free(arr_t3456);
    free(arr_t7);
    free(arr_t8);

    //CheckError(pgm_destroy_graph(g));
    CheckError(pgm_destroy());
    fprintf(stdout, "cleaned up ...");
}

void App::sched_ABCDE_hog(cv::Ptr<cv::cuda::HOG> gpu_hog, cv::HOGDescriptor cpu_hog, Mat* frames)
{
    pthread_barrier_t arr_fine_init_barrier[args.num_fine_graphs];
    /* graph construction */
    graph_t arr_g [args.num_fine_graphs];

    struct sync_info arr_sync_info_ABCDE [args.num_fine_graphs][NUM_SCALE_LEVELS];

    thread** arr_t0     = (thread**) calloc(args.num_fine_graphs, sizeof(std::thread *));
    thread** arr_t1     = (thread**) calloc(args.num_fine_graphs, sizeof(std::thread *));
    thread** arr_t23456 = (thread**) calloc(args.num_fine_graphs * NUM_SCALE_LEVELS, sizeof(std::thread *));
    thread** arr_t7     = (thread**) calloc(args.num_fine_graphs, sizeof(std::thread *));
    thread** arr_t8     = (thread**) calloc(args.num_fine_graphs, sizeof(std::thread *));

    char buf[30];
    sprintf(buf, "/tmp/graph_t%d", args.task_id);
    CheckError(pgm_init(buf, 1));

    for (int g_idx = 0; g_idx < args.num_fine_graphs; g_idx++)
    {
        pthread_barrier_t* fine_init_barrier = arr_fine_init_barrier + g_idx;

        // A graph consists of the graph itself, nodes, and edge
        graph_t* g_ptr = arr_g + g_idx;

        node_t color_convert_node;
        node_t compute_scales_node;
        node_t ABCDE_node              [NUM_SCALE_LEVELS]; // resize -> classify-hists
        node_t collect_locations_node;
        node_t display_node;

        edge_t e0_1;
        edge_t e1_23456 [NUM_SCALE_LEVELS];
        edge_t e23456_7 [NUM_SCALE_LEVELS];
        edge_t e7_8;

        // Initialize the graph
        sprintf(buf, "hog_%d", g_idx);
        CheckError(pgm_init_graph(g_ptr, buf));
        graph_t g = *g_ptr;

        // Initialize the nodes
        CheckError(pgm_init_node(&color_convert_node, g, "color_convert"));
        CheckError(pgm_init_node(&compute_scales_node, g, "compute_scales"));
        for (int i=0; i<NUM_SCALE_LEVELS; i++) {
            CheckError(pgm_init_node(ABCDE_node + i, g, "ABCDE"));
        }
        CheckError(pgm_init_node(&collect_locations_node, g, "collect_locations"));
        CheckError(pgm_init_node(&display_node, g, "display"));

        // Initialize the edges
        edge_attr_t fast_mq_attr;
        memset(&fast_mq_attr, 0, sizeof(fast_mq_attr));
        fast_mq_attr.type = pgm_fast_fifo_edge;

        fast_mq_attr.nr_produce = sizeof(struct params_compute);
        fast_mq_attr.nr_consume = sizeof(struct params_compute);
        fast_mq_attr.nr_threshold = sizeof(struct params_compute);
        CheckError(pgm_init_edge(&e0_1, color_convert_node, compute_scales_node, "e0_1", &fast_mq_attr));

        for (int i=0; i<NUM_SCALE_LEVELS; i++) {
            sprintf(buf, "e1_23456_%d", i);
            fast_mq_attr.nr_produce = sizeof(struct params_resize);
            fast_mq_attr.nr_consume = sizeof(struct params_resize);
            fast_mq_attr.nr_threshold = sizeof(struct params_resize);
            CheckError(pgm_init_edge(e1_23456 + i,
                        compute_scales_node,
                        ABCDE_node[i], buf, &fast_mq_attr));

            sprintf(buf, "e23456_7_%d", i);
            fast_mq_attr.nr_produce = sizeof(struct params_fine_collect_locations);
            fast_mq_attr.nr_consume = sizeof(struct params_fine_collect_locations);
            fast_mq_attr.nr_threshold = sizeof(struct params_fine_collect_locations);
            CheckError(pgm_init_edge(e23456_7 + i,
                        ABCDE_node[i],
                        collect_locations_node, buf, &fast_mq_attr));
        }

        fast_mq_attr.nr_produce = sizeof(struct params_display);
        fast_mq_attr.nr_consume = sizeof(struct params_display);
        fast_mq_attr.nr_threshold = sizeof(struct params_display);
        CheckError(pgm_init_edge(&e7_8, collect_locations_node, display_node, "e7_8", &fast_mq_attr));

        pthread_barrier_init(fine_init_barrier, 0, 1 * NUM_SCALE_LEVELS + 4);

        thread** t0  = arr_t0 + g_idx;
        thread** t1  = arr_t1 + g_idx;
        thread** t23456 = arr_t23456 + g_idx * NUM_SCALE_LEVELS;
        thread** t7  = arr_t7 + g_idx;
        thread** t8  = arr_t8 + g_idx;

        struct sync_info* in_sync_info_ABCDE              = arr_sync_info_ABCDE             [((g_idx + args.num_fine_graphs - 1) % args.num_fine_graphs)];

        struct sync_info* out_sync_info_ABCDE              = arr_sync_info_ABCDE             [g_idx];

        for (int i=0; i<NUM_SCALE_LEVELS; i++) {
            sync_info_init(i + in_sync_info_ABCDE);

            sync_info_init(i + out_sync_info_ABCDE);
        }

        /* G-FL fine-grained HOG */
        float bound_color_convert                   = 31.0194960215;
        float bound_compute_scales                  = 30.5688293615;
        // float bounds_resize      [NUM_SCALE_LEVELS] = {30.2419972231,30.3148010256,30.2936775569,30.3079650324,30.2963991483,30.3058228698,30.3005336032,30.2888197835,30.293491626,30.2840584057,30.3013304497,30.3016323015,30.3111116087};
        // float bounds_compute_grad[NUM_SCALE_LEVELS] = {30.4902863291,30.44552396,30.4706995604,30.4291495557,30.4171804989,30.4135009326,30.3785928987,30.4059076714,30.3816881109,30.3627509567,30.3564101342,30.3519205285,30.3405616838};
        // float bounds_compute_hist[NUM_SCALE_LEVELS] = {30.5326861275,30.4951636278,30.5415047381,30.4958614402,30.5149987735,30.4750125581,30.5373623676,30.522424222,30.4851937235,30.4577396346,30.4263855713,30.4118414514,30.5202006157};
        float bounds_ABCDE       [NUM_SCALE_LEVELS] = {30.2419972231,30.3148010256,30.2936775569,30.3079650324,30.2963991483,30.3058228698,30.3005336032,30.2888197835,30.293491626,30.2840584057,30.3013304497,30.3016323015,30.3111116087};
        // float bounds_normalize   [NUM_SCALE_LEVELS] = {30.2903383654,30.3080869341,30.2928467608,30.3064130287,30.3114078316,30.3120987837,30.2985990076,30.2803770804,30.3099685825,30.3123524377,30.2989110618,30.2943410679,30.3081075149};
        // float bounds_classify    [NUM_SCALE_LEVELS] = {30.4792191337,30.4948973087,30.4540269983,30.429525815,30.5229441951,30.5070612675,30.497249677,30.4955912512,30.4267037822,30.4508744413,30.430038752,30.3991297266,30.3767439685};
        //float bound_collect_locations               30.983976896;

        /* G-FL fine-grained HOG */
        float cost_color_convert                   = 4.509346;
        float cost_compute_scales                  = 1.947347;
        // float costs_resize      [NUM_SCALE_LEVELS] = {0.089336,0.503219,0.383134,0.464357,0.398606,0.452179,0.42211,0.355518,0.382077,0.32845,0.42664,0.428356,0.482245};
        // float costs_compute_grad[NUM_SCALE_LEVELS] = {1.500837,1.246367,1.389488,1.15328,1.085237,1.064319,0.86587,1.021152,0.883466,0.77581,0.739763,0.71424,0.649666};
        // float costs_compute_hist[NUM_SCALE_LEVELS] = {1.741876,1.528564,1.792009,1.532531,1.641325,1.414007,1.76846,1.683538,1.471886,1.315812,1.137567,1.054885,1.670897};
        float costs_ABCDE       [NUM_SCALE_LEVELS] = {0.089336,0.503219,0.383134,0.464357,0.398606,0.452179,0.42211,0.355518,0.382077,0.32845,0.42664,0.428356,0.482245};
        // float costs_normalize   [NUM_SCALE_LEVELS] = {0.364151,0.46505,0.378411,0.455534,0.483929,0.487857,0.411112,0.307522,0.475747,0.489299,0.412886,0.386906,0.465167};
        // float costs_classify    [NUM_SCALE_LEVELS] = {1.437921,1.52705,1.294706,1.155419,1.686494,1.596201,1.540423,1.530995,1.139376,1.276784,1.158335,0.98262,0.855359};
        float cost_collect_locations               = 4.307423;

        /* C-FL fine-grained HOG */
        //float bound_color_convert                   = 26.3496352098;
        //float bound_compute_scales                  = 25.6841982494;
        //float bounds_resize      [NUM_SCALE_LEVELS] = {25.0939476981,25.1387935171,25.130189875,25.1357033275,25.1312621102,25.1463332249,25.1316714257,25.1302245094,25.1267072576,25.1262401226,25.1348909935,25.1432006726,25.1413123823};
        //float bounds_compute_grad[NUM_SCALE_LEVELS] = {25.3489965402,25.4185896374,25.4092134479,25.4079932294,25.4051932818,25.4030496702,25.312479578,25.291207475,25.2868561354,25.2685316772,25.2518075571,25.246440943,25.2364255915};
        //float bounds_compute_hist[NUM_SCALE_LEVELS] = {25.5728014354,25.4973631396,25.5123077387,25.471628638,25.3841716342,25.3274278981,25.4720253593,25.5328196023,25.4654399579,25.3426074958,25.4860697502,25.3209827517,25.4610373822};
        //float bounds_ABCDE       [NUM_SCALE_LEVELS] = {25.0939476981,25.1387935171,25.130189875,25.1357033275,25.1312621102,25.1463332249,25.1316714257,25.1302245094,25.1267072576,25.1262401226,25.1348909935,25.1432006726,25.1413123823};
        //float bounds_normalize   [NUM_SCALE_LEVELS] = {25.1658996554,25.1786894798,25.1487118351,25.1680595823,25.1750299693,25.1885187788,25.1646279147,25.1649092834,25.1582577618,25.1765865135,25.1940356661,25.1748711091,25.179600851};
        //float bounds_classify    [NUM_SCALE_LEVELS] = {25.2909527262,25.3017792668,25.4537335318,25.3216511097,25.4693284561,25.3272781973,25.3334016159,25.3265829333,25.3061011243,25.2757711248,25.2811148401,25.2748319888,25.3276382806};
        ////float bound_collect_locations               26.204139258;

        /////* C-FL fine-grained HOG */
        //float cost_color_convert                   = 4.475663;
        //float cost_compute_scales                  = 2.150868;
        //float costs_resize      [NUM_SCALE_LEVELS] = {0.088747,0.245422,0.215364,0.234626,0.21911,0.271763,0.22054,0.215485,0.203197,0.201565,0.231788,0.260819,0.254222};
        //float costs_compute_grad[NUM_SCALE_LEVELS] = {0.979795,1.222928,1.190171,1.185908,1.176126,1.168637,0.852218,0.777901,0.762699,0.69868,0.640252,0.621503,0.586513};
        //float costs_compute_hist[NUM_SCALE_LEVELS] = {1.761688,1.498134,1.550345,1.408227,1.102684,0.904442,1.409613,1.622006,1.386606,0.957474,1.458679,0.881925,1.371225};
        //float costs_ABCDE       [NUM_SCALE_LEVELS] = {0.088747,0.245422,0.215364,0.234626,0.21911,0.271763,0.22054,0.215485,0.203197,0.201565,0.231788,0.260819,0.254222};
        //float costs_normalize   [NUM_SCALE_LEVELS] = {0.340121,0.384804,0.280073,0.347667,0.372019,0.419144,0.335678,0.336661,0.313423,0.377457,0.438418,0.371464,0.387988};
        //float costs_classify    [NUM_SCALE_LEVELS] = {0.777011,0.814835,1.345708,0.88426,1.400191,0.903919,0.925312,0.90149,0.829934,0.723972,0.742641,0.720691,0.905177};
        //float cost_collect_locations               = 3.967353;

        /* | first graph release      | second graph release     | first graph release again
         *  <---------PERIOD--------->
         *  <--------------- PERIOD * args.num_fine_graphs ---------->
         */
        int period = PERIOD * args.num_fine_graphs;
        int m_cpus = 16;
        struct task_info t_info;
        t_info.early = args.early;
        t_info.realtime = args.realtime;
        t_info.sched = fine_ABCDE;
        t_info.period = period;
        t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, cost_color_convert);
        t_info.phase = PERIOD * g_idx;
        t_info.id = 0;
        if (args.cluster != -1)
            t_info.cluster = args.cluster;
        else
            t_info.cluster = args.cluster;
        *t0 = new thread(&App::thread_color_convert, this,
                &color_convert_node, fine_init_barrier,
                gpu_hog, cpu_hog, frames, t_info, g_idx);

        t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, cost_compute_scales);
        t_info.id = 1;
        t_info.phase = t_info.phase + bound_color_convert;
        *t1 = new thread(&cv::cuda::HOG::thread_fine_compute_scales, gpu_hog,
                &compute_scales_node, fine_init_barrier, t_info);

        for (int i=0; i<NUM_SCALE_LEVELS; i++) {
            t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, costs_ABCDE[i]);
            t_info.id = 0 * NUM_SCALE_LEVELS + i + 2;
            t_info.phase = PERIOD * g_idx + bound_color_convert + bound_compute_scales;
            t_info.s_info_in = in_sync_info_ABCDE + i;
            t_info.s_info_out = out_sync_info_ABCDE + i;
            t23456[i] = new thread(&cv::cuda::HOG::thread_fine_ABCDE, gpu_hog,
                    ABCDE_node + i, fine_init_barrier, t_info);
        }

        t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, cost_collect_locations);
        t_info.id = 1 * NUM_SCALE_LEVELS + 2;
        t_info.phase = t_info.phase + *std::max_element(bounds_ABCDE, bounds_ABCDE+NUM_SCALE_LEVELS);
        *t7 = new thread(&cv::cuda::HOG::thread_fine_collect_locations, gpu_hog,
                &collect_locations_node, fine_init_barrier, t_info);
        //t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, cost_display);
        //t_info.phase;
        *t8 = new thread(&App::thread_display, this, &display_node,
                fine_init_barrier, g_idx == 0 && args.display);
    }

    /* graph construction finishes */

    printf("Joining pthreads...\n");

    for (int g_idx = 0; g_idx < args.num_fine_graphs; g_idx++) {
        graph_t g = arr_g[g_idx];
        thread* t0 = arr_t0[g_idx];
        thread* t1 = arr_t1[g_idx];
        thread** t23456 = arr_t23456 + g_idx * NUM_SCALE_LEVELS;
        thread* t7 = arr_t7[g_idx];
        thread* t8 = arr_t8[g_idx];
        t0->join();
        t1->join();
        delete t0;
        delete t1;
        for (int i=0; i<NUM_SCALE_LEVELS; i++) {
            if (t23456[i]->joinable()) t23456[i]->join();

            delete t23456[i];
        }
        t7->join();
        t8->join();
        delete t7;
        delete t8;
        CheckError(pgm_destroy_graph(g));
    }
    free(arr_t1);
    free(arr_t23456);
    free(arr_t7);
    free(arr_t8);

    //CheckError(pgm_destroy_graph(g));
    CheckError(pgm_destroy());
    fprintf(stdout, "cleaned up ...");
}

void App::sched_single_merge_in_level_hog(cv::Ptr<cv::cuda::HOG> gpu_hog, cv::HOGDescriptor cpu_hog, Mat* frames)
{
    // Each level has at most one merged node
    scheduling_option level_options[13] = {
        fine_grained, /* level  0 */
        fine_grained, /* level  1 */
        fine_AB,      /* level  2 */
        fine_BC,      /* level  3 */
        fine_CD,      /* level  4 */
        fine_DE,      /* level  5 */
        fine_ABC,     /* level  6 */
        fine_BCD,     /* level  7 */
        fine_CDE,     /* level  8 */
        fine_ABCD,    /* level  9 */
        fine_BCDE,    /* level 10 */
        fine_ABCDE,   /* level 11 */
        fine_ABCDE    /* level 12 */
    };

    // Gather statistics about the levels
    unsigned num_nodes_per_level[13] = {
        5,
        5,
        4,
        4,
        4,
        4,
        3,
        3,
        3,
        2,
        2,
        1,
        1
    };

    int num_total_level_nodes = 0;
    for (unsigned i = 0; i < NUM_SCALE_LEVELS; i++)
    {
        num_total_level_nodes += num_nodes_per_level[i];
    }

    pthread_barrier_t arr_fine_init_barrier[args.num_fine_graphs];
    /* graph construction */
    graph_t arr_g [args.num_fine_graphs];

    // Not all sync info structs will be used
    struct sync_info arr_sync_info_1st [args.num_fine_graphs][NUM_SCALE_LEVELS];
    struct sync_info arr_sync_info_2nd [args.num_fine_graphs][NUM_SCALE_LEVELS];
    struct sync_info arr_sync_info_3rd [args.num_fine_graphs][NUM_SCALE_LEVELS];
    struct sync_info arr_sync_info_4th [args.num_fine_graphs][NUM_SCALE_LEVELS];
    struct sync_info arr_sync_info_5th [args.num_fine_graphs][NUM_SCALE_LEVELS];

    thread** arr_t0  = (thread**) calloc(args.num_fine_graphs, sizeof(std::thread *));
    thread** arr_t1  = (thread**) calloc(args.num_fine_graphs, sizeof(std::thread *));
    thread** arr_t1st = (thread**) calloc(args.num_fine_graphs * NUM_SCALE_LEVELS, sizeof(std::thread *));
    thread** arr_t2nd = (thread**) calloc(args.num_fine_graphs * NUM_SCALE_LEVELS, sizeof(std::thread *));
    thread** arr_t3rd = (thread**) calloc(args.num_fine_graphs * NUM_SCALE_LEVELS, sizeof(std::thread *));
    thread** arr_t4th = (thread**) calloc(args.num_fine_graphs * NUM_SCALE_LEVELS, sizeof(std::thread *));
    thread** arr_t5th = (thread**) calloc(args.num_fine_graphs * NUM_SCALE_LEVELS, sizeof(std::thread *));
    thread** arr_t7  = (thread**) calloc(args.num_fine_graphs, sizeof(std::thread *));
    thread** arr_t8  = (thread**) calloc(args.num_fine_graphs, sizeof(std::thread *));

    char buf[30];
    sprintf(buf, "/tmp/graph_t%d", args.task_id);
    CheckError(pgm_init(buf, 1));

    for (int g_idx = 0; g_idx < args.num_fine_graphs; g_idx++)
    {
        pthread_barrier_t* fine_init_barrier = arr_fine_init_barrier + g_idx;

        // A graph consists of the graph itself, nodes, and edge
        graph_t* g_ptr = arr_g + g_idx;

        node_t color_convert_node;
        node_t compute_scales_node;
        node_t level_nodes [NUM_SCALE_LEVELS][5];
        node_t collect_locations_node;
        node_t display_node;

        edge_t e0_1;
        edge_t level_edges [NUM_SCALE_LEVELS][6];
        edge_t e7_8;

        // Initialize the graph
        sprintf(buf, "hog_%d", g_idx);
        CheckError(pgm_init_graph(g_ptr, buf));
        graph_t g = *g_ptr;

        // Initialize the nodes
        const char* level_node_names[] = { "node_1st", "node_2nd", "node_3rd", "node_4th", "node_5th" };
        CheckError(pgm_init_node(&color_convert_node, g, "color_convert"));
        CheckError(pgm_init_node(&compute_scales_node, g, "compute_scales"));
        for (int i=0; i<NUM_SCALE_LEVELS; i++) {
            unsigned int num_nodes = num_nodes_per_level[i];
            for (unsigned node_idx = 0; node_idx < num_nodes; node_idx++)
            {
                CheckError(pgm_init_node(&(level_nodes[i][node_idx]), g, level_node_names[node_idx]));
            }
        }
        CheckError(pgm_init_node(&collect_locations_node, g, "collect_locations"));
        CheckError(pgm_init_node(&display_node, g, "display"));

        // Initialize the edges
        const char* level_edge_name_formats[] = { "e1_1st_%d", "e1st_2nd_%d", "e2nd_3rd_%d", "e3rd_4th_%d", "e4th_5th_%d", "e5th_7_%d" };
        edge_attr_t fast_mq_attr;
        memset(&fast_mq_attr, 0, sizeof(fast_mq_attr));
        fast_mq_attr.type = pgm_fast_fifo_edge;

        fast_mq_attr.nr_produce = sizeof(struct params_compute);
        fast_mq_attr.nr_consume = sizeof(struct params_compute);
        fast_mq_attr.nr_threshold = sizeof(struct params_compute);
        CheckError(pgm_init_edge(&e0_1, color_convert_node, compute_scales_node, "e0_1", &fast_mq_attr));

        for (int i=0; i<NUM_SCALE_LEVELS; i++) {
            size_t params_sizes[num_nodes_per_level[i]];

            switch (level_options[i]) {
                case fine_grained:
                    params_sizes[0] = sizeof(struct params_resize);
                    params_sizes[1] = sizeof(struct params_compute_gradients);
                    params_sizes[2] = sizeof(struct params_compute_histograms);
                    params_sizes[3] = sizeof(struct params_fine_normalize);
                    params_sizes[4] = sizeof(struct params_fine_classify);
                    params_sizes[5] = sizeof(struct params_fine_collect_locations);
                    break;
                case fine_AB:
                    params_sizes[0] = sizeof(struct params_resize);
                    params_sizes[1] = sizeof(struct params_compute_histograms);
                    params_sizes[2] = sizeof(struct params_fine_normalize);
                    params_sizes[3] = sizeof(struct params_fine_classify);
                    params_sizes[4] = sizeof(struct params_fine_collect_locations);
                    break;
                case fine_BC:
                    params_sizes[0] = sizeof(struct params_resize);
                    params_sizes[1] = sizeof(struct params_compute_gradients);
                    params_sizes[2] = sizeof(struct params_fine_normalize);
                    params_sizes[3] = sizeof(struct params_fine_classify);
                    params_sizes[4] = sizeof(struct params_fine_collect_locations);
                    break;
                case fine_CD:
                    params_sizes[0] = sizeof(struct params_resize);
                    params_sizes[1] = sizeof(struct params_compute_gradients);
                    params_sizes[2] = sizeof(struct params_compute_histograms);
                    params_sizes[3] = sizeof(struct params_fine_classify);
                    params_sizes[4] = sizeof(struct params_fine_collect_locations);
                    break;
                case fine_DE:
                    params_sizes[0] = sizeof(struct params_resize);
                    params_sizes[1] = sizeof(struct params_compute_gradients);
                    params_sizes[2] = sizeof(struct params_compute_histograms);
                    params_sizes[3] = sizeof(struct params_fine_normalize);
                    params_sizes[4] = sizeof(struct params_fine_collect_locations);
                    break;
                case fine_ABC:
                    params_sizes[0] = sizeof(struct params_resize);
                    params_sizes[1] = sizeof(struct params_fine_normalize);
                    params_sizes[2] = sizeof(struct params_fine_classify);
                    params_sizes[3] = sizeof(struct params_fine_collect_locations);
                    break;
                case fine_BCD:
                    params_sizes[0] = sizeof(struct params_resize);
                    params_sizes[1] = sizeof(struct params_compute_gradients);
                    params_sizes[2] = sizeof(struct params_fine_classify);
                    params_sizes[3] = sizeof(struct params_fine_collect_locations);
                    break;
                case fine_CDE:
                    params_sizes[0] = sizeof(struct params_resize);
                    params_sizes[1] = sizeof(struct params_compute_gradients);
                    params_sizes[2] = sizeof(struct params_compute_histograms);
                    params_sizes[3] = sizeof(struct params_fine_collect_locations);
                    break;
                case fine_ABCD:
                    params_sizes[0] = sizeof(struct params_resize);
                    params_sizes[1] = sizeof(struct params_fine_classify);
                    params_sizes[2] = sizeof(struct params_fine_collect_locations);
                    break;
                case fine_BCDE:
                    params_sizes[0] = sizeof(struct params_resize);
                    params_sizes[1] = sizeof(struct params_compute_gradients);
                    params_sizes[2] = sizeof(struct params_fine_collect_locations);
                    break;
                case fine_ABCDE:
                    params_sizes[0] = sizeof(struct params_resize);
                    params_sizes[1] = sizeof(struct params_fine_collect_locations);
                    break;
                default:
                    break;
            }

            for (unsigned edge_idx = 0; edge_idx < num_nodes_per_level[i]+1; edge_idx++)
            {
                // Edge parameters: name, token counts
                sprintf(buf, level_edge_name_formats[edge_idx], i);
                fast_mq_attr.nr_produce = params_sizes[edge_idx];
                fast_mq_attr.nr_consume = params_sizes[edge_idx];
                fast_mq_attr.nr_threshold = params_sizes[edge_idx];

                // Choose the nodes connected by the edge and initialize the edge
                node_t node_start = edge_idx == 0                      ? compute_scales_node    : level_nodes[i][edge_idx-1];
                node_t node_end   = edge_idx == num_nodes_per_level[i] ? collect_locations_node : level_nodes[i][edge_idx];
                CheckError(pgm_init_edge(&(level_edges[i][edge_idx]),
                                            node_start, node_end,
                                            buf, &fast_mq_attr));
            }
        }

        fast_mq_attr.nr_produce = sizeof(struct params_display);
        fast_mq_attr.nr_consume = sizeof(struct params_display);
        fast_mq_attr.nr_threshold = sizeof(struct params_display);
        CheckError(pgm_init_edge(&e7_8, collect_locations_node, display_node, "e7_8", &fast_mq_attr));

        pthread_barrier_init(fine_init_barrier, 0, num_total_level_nodes + 4);

        thread** t0   = arr_t0 + g_idx;
        thread** t1   = arr_t1 + g_idx;
        thread** t1st = arr_t1st + g_idx * NUM_SCALE_LEVELS;
        thread** t2nd = arr_t2nd + g_idx * NUM_SCALE_LEVELS;
        thread** t3rd = arr_t3rd + g_idx * NUM_SCALE_LEVELS;
        thread** t4th = arr_t4th + g_idx * NUM_SCALE_LEVELS;
        thread** t5th = arr_t5th + g_idx * NUM_SCALE_LEVELS;
        thread** t7   = arr_t7 + g_idx;
        thread** t8   = arr_t8 + g_idx;

        struct sync_info* in_sync_info_1st = arr_sync_info_1st[((g_idx + args.num_fine_graphs - 1) % args.num_fine_graphs)];
        struct sync_info* in_sync_info_2nd = arr_sync_info_2nd[((g_idx + args.num_fine_graphs - 1) % args.num_fine_graphs)];
        struct sync_info* in_sync_info_3rd = arr_sync_info_3rd[((g_idx + args.num_fine_graphs - 1) % args.num_fine_graphs)];
        struct sync_info* in_sync_info_4th = arr_sync_info_4th[((g_idx + args.num_fine_graphs - 1) % args.num_fine_graphs)];
        struct sync_info* in_sync_info_5th = arr_sync_info_5th[((g_idx + args.num_fine_graphs - 1) % args.num_fine_graphs)];

        struct sync_info* out_sync_info_1st = arr_sync_info_1st[g_idx];
        struct sync_info* out_sync_info_2nd = arr_sync_info_2nd[g_idx];
        struct sync_info* out_sync_info_3rd = arr_sync_info_3rd[g_idx];
        struct sync_info* out_sync_info_4th = arr_sync_info_4th[g_idx];
        struct sync_info* out_sync_info_5th = arr_sync_info_5th[g_idx];

        for (int i=0; i<NUM_SCALE_LEVELS; i++) {
            switch (level_options[i]) {
                case fine_grained:
                    sync_info_init(i + in_sync_info_1st);
                    sync_info_init(i + in_sync_info_2nd);
                    sync_info_init(i + in_sync_info_3rd);
                    sync_info_init(i + in_sync_info_4th);
                    sync_info_init(i + in_sync_info_5th);

                    sync_info_init(i + out_sync_info_1st);
                    sync_info_init(i + out_sync_info_2nd);
                    sync_info_init(i + out_sync_info_3rd);
                    sync_info_init(i + out_sync_info_4th);
                    sync_info_init(i + out_sync_info_5th);
                    break;
                case fine_AB:
                case fine_BC:
                case fine_CD:
                case fine_DE:
                    sync_info_init(i + in_sync_info_1st);
                    sync_info_init(i + in_sync_info_2nd);
                    sync_info_init(i + in_sync_info_3rd);
                    sync_info_init(i + in_sync_info_4th);

                    sync_info_init(i + out_sync_info_1st);
                    sync_info_init(i + out_sync_info_2nd);
                    sync_info_init(i + out_sync_info_3rd);
                    sync_info_init(i + out_sync_info_4th);
                    break;
                case fine_ABC:
                case fine_BCD:
                case fine_CDE:
                    sync_info_init(i + in_sync_info_1st);
                    sync_info_init(i + in_sync_info_2nd);
                    sync_info_init(i + in_sync_info_3rd);

                    sync_info_init(i + out_sync_info_1st);
                    sync_info_init(i + out_sync_info_2nd);
                    sync_info_init(i + out_sync_info_3rd);
                    break;
                case fine_ABCD:
                case fine_BCDE:
                    sync_info_init(i + in_sync_info_1st);
                    sync_info_init(i + in_sync_info_2nd);

                    sync_info_init(i + out_sync_info_1st);
                    sync_info_init(i + out_sync_info_2nd);
                    break;
                case fine_ABCDE:
                    sync_info_init(i + in_sync_info_1st);

                    sync_info_init(i + out_sync_info_1st);
                    break;
                default:
                    break;
            }
        }

        float bound_color_convert           = 31.0194960215;
        float bound_compute_scales          = 30.5688293615;
        float bounds_1st [NUM_SCALE_LEVELS] = {30.2419972231,30.3148010256,30.2936775569,30.3079650324,30.2963991483,30.3058228698,30.3005336032,30.2888197835,30.293491626,30.2840584057,30.3013304497,30.3016323015,30.3111116087};
        float bounds_2nd [NUM_SCALE_LEVELS] = {30.4902863291,30.44552396,30.4706995604,30.4291495557,30.4171804989,30.4135009326,30.3785928987,30.4059076714,30.3816881109,30.3627509567,30.3564101342,30.3519205285,30.3405616838};
        float bounds_3rd [NUM_SCALE_LEVELS] = {30.5326861275,30.4951636278,30.5415047381,30.4958614402,30.5149987735,30.4750125581,30.5373623676,30.522424222,30.4851937235,30.4577396346,30.4263855713,30.4118414514,30.5202006157};
        float bounds_4th [NUM_SCALE_LEVELS] = {30.2903383654,30.3080869341,30.2928467608,30.3064130287,30.3114078316,30.3120987837,30.2985990076,30.2803770804,30.3099685825,30.3123524377,30.2989110618,30.2943410679,30.3081075149};
        float bounds_5th [NUM_SCALE_LEVELS] = {30.4792191337,30.4948973087,30.4540269983,30.429525815,30.5229441951,30.5070612675,30.497249677,30.4955912512,30.4267037822,30.4508744413,30.430038752,30.3991297266,30.3767439685};

        float cost_color_convert           = 4.509346;
        float cost_compute_scales          = 1.947347;
        float costs_1st [NUM_SCALE_LEVELS] = {0.089336,0.503219,0.383134,0.464357,0.398606,0.452179,0.42211,0.355518,0.382077,0.32845,0.42664,0.428356,0.482245};
        float costs_2nd [NUM_SCALE_LEVELS] = {1.500837,1.246367,1.389488,1.15328,1.085237,1.064319,0.86587,1.021152,0.883466,0.77581,0.739763,0.71424,0.649666};
        float costs_3rd [NUM_SCALE_LEVELS] = {1.741876,1.528564,1.792009,1.532531,1.641325,1.414007,1.76846,1.683538,1.471886,1.315812,1.137567,1.054885,1.670897};
        float costs_4th [NUM_SCALE_LEVELS] = {0.364151,0.46505,0.378411,0.455534,0.483929,0.487857,0.411112,0.307522,0.475747,0.489299,0.412886,0.386906,0.465167};
        float costs_5th [NUM_SCALE_LEVELS] = {1.437921,1.52705,1.294706,1.155419,1.686494,1.596201,1.540423,1.530995,1.139376,1.276784,1.158335,0.98262,0.855359};
        float cost_collect_locations       = 4.307423;

        /* | first graph release      | second graph release     | first graph release again
         *  <---------PERIOD--------->
         *  <--------------- PERIOD * args.num_fine_graphs ---------->
         */
        unsigned task_id = 0;
        int period = PERIOD * args.num_fine_graphs;
        int m_cpus = 16;
        struct task_info t_info;
        t_info.early = args.early;
        t_info.realtime = args.realtime;
        t_info.sched = fine_AB;
        t_info.period = period;
        t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, cost_color_convert);
        t_info.phase = PERIOD * g_idx;
        t_info.id = task_id++;
        if (args.cluster != -1)
            t_info.cluster = args.cluster;
        else
            t_info.cluster = args.cluster;
        *t0 = new thread(&App::thread_color_convert, this,
                &color_convert_node, fine_init_barrier,
                gpu_hog, cpu_hog, frames, t_info, g_idx);

        t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, cost_compute_scales);
        t_info.id = task_id++;
        t_info.phase = t_info.phase + bound_color_convert;
        *t1 = new thread(&cv::cuda::HOG::thread_fine_compute_scales, gpu_hog,
                &compute_scales_node, fine_init_barrier, t_info);

        for (int i=0; i<NUM_SCALE_LEVELS; i++) {
            switch (level_options[i]) {
                case fine_grained:
                    t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, costs_1st[i]);
                    t_info.id = task_id++;
                    t_info.phase = PERIOD * g_idx + bound_color_convert + bound_compute_scales;
                    t_info.s_info_in = in_sync_info_1st + i;
                    t_info.s_info_out = out_sync_info_1st + i;
                    t1st[i] = new thread(&cv::cuda::HOG::thread_fine_resize, gpu_hog,
                            &(level_nodes[i][0]), fine_init_barrier, t_info);

                    t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, costs_2nd[i]);
                    t_info.id = task_id++;
                    t_info.phase = t_info.phase + bounds_1st[i];
                    t_info.s_info_in = in_sync_info_2nd + i;
                    t_info.s_info_out = out_sync_info_2nd + i;
                    t2nd[i] = new thread(&cv::cuda::HOG::thread_fine_compute_gradients,
                            gpu_hog, &(level_nodes[i][1]), fine_init_barrier,
                            t_info);

                    t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, costs_3rd[i]);
                    t_info.id = task_id++;
                    t_info.phase = t_info.phase + bounds_2nd[i];
                    t_info.s_info_in = in_sync_info_3rd + i;
                    t_info.s_info_out = out_sync_info_3rd + i;
                    t3rd[i] = new thread(&cv::cuda::HOG::thread_fine_compute_histograms,
                            gpu_hog, &(level_nodes[i][2]), fine_init_barrier, t_info);

                    t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, costs_4th[i]);
                    t_info.id = task_id++;
                    t_info.phase = t_info.phase + bounds_3rd[i];
                    t_info.s_info_in = in_sync_info_4th + i;
                    t_info.s_info_out = out_sync_info_4th + i;
                    t4th[i] = new thread(&cv::cuda::HOG::thread_fine_normalize_histograms, gpu_hog,
                            &(level_nodes[i][3]), fine_init_barrier, t_info);

                    t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, costs_5th[i]);
                    t_info.id = task_id++;
                    t_info.phase = t_info.phase + bounds_4th[i];
                    t_info.s_info_in = in_sync_info_5th + i;
                    t_info.s_info_out = out_sync_info_5th + i;
                    t5th[i] = new thread(&cv::cuda::HOG::thread_fine_classify, gpu_hog,
                            &(level_nodes[i][4]), fine_init_barrier, t_info);
                    break;
                case fine_AB:
                    t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, costs_1st[i]);
                    t_info.id = task_id++;
                    t_info.phase = PERIOD * g_idx + bound_color_convert + bound_compute_scales;
                    t_info.s_info_in = in_sync_info_1st + i;
                    t_info.s_info_out = out_sync_info_1st + i;
                    t1st[i] = new thread(&cv::cuda::HOG::thread_fine_AB, gpu_hog,
                            &(level_nodes[i][0]), fine_init_barrier, t_info);

                    t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, costs_2nd[i]);
                    t_info.id = task_id++;
                    t_info.phase = t_info.phase + bounds_1st[i];
                    t_info.s_info_in = in_sync_info_2nd + i;
                    t_info.s_info_out = out_sync_info_2nd + i;
                    t2nd[i] = new thread(&cv::cuda::HOG::thread_fine_compute_histograms,
                            gpu_hog, &(level_nodes[i][1]), fine_init_barrier,
                            t_info);

                    t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, costs_3rd[i]);
                    t_info.id = task_id++;
                    t_info.phase = t_info.phase + bounds_2nd[i];
                    t_info.s_info_in = in_sync_info_3rd + i;
                    t_info.s_info_out = out_sync_info_3rd + i;
                    t3rd[i] = new thread(&cv::cuda::HOG::thread_fine_normalize_histograms,
                            gpu_hog, &(level_nodes[i][2]), fine_init_barrier, t_info);

                    t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, costs_4th[i]);
                    t_info.id = task_id++;
                    t_info.phase = t_info.phase + bounds_3rd[i];
                    t_info.s_info_in = in_sync_info_4th + i;
                    t_info.s_info_out = out_sync_info_4th + i;
                    t4th[i] = new thread(&cv::cuda::HOG::thread_fine_classify, gpu_hog,
                            &(level_nodes[i][3]), fine_init_barrier, t_info);
                    break;
                case fine_BC:
                    t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, costs_1st[i]);
                    t_info.id = task_id++;
                    t_info.phase = PERIOD * g_idx + bound_color_convert + bound_compute_scales;
                    t_info.s_info_in = in_sync_info_1st + i;
                    t_info.s_info_out = out_sync_info_1st + i;
                    t1st[i] = new thread(&cv::cuda::HOG::thread_fine_resize, gpu_hog,
                            &(level_nodes[i][0]), fine_init_barrier, t_info);

                    t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, costs_2nd[i]);
                    t_info.id = task_id++;
                    t_info.phase = t_info.phase + bounds_1st[i];
                    t_info.s_info_in = in_sync_info_2nd + i;
                    t_info.s_info_out = out_sync_info_2nd + i;
                    t2nd[i] = new thread(&cv::cuda::HOG::thread_fine_BC,
                            gpu_hog, &(level_nodes[i][1]), fine_init_barrier,
                            t_info);

                    t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, costs_3rd[i]);
                    t_info.id = task_id++;
                    t_info.phase = t_info.phase + bounds_2nd[i];
                    t_info.s_info_in = in_sync_info_3rd + i;
                    t_info.s_info_out = out_sync_info_3rd + i;
                    t3rd[i] = new thread(&cv::cuda::HOG::thread_fine_normalize_histograms,
                            gpu_hog, &(level_nodes[i][2]), fine_init_barrier, t_info);

                    t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, costs_4th[i]);
                    t_info.id = task_id++;
                    t_info.phase = t_info.phase + bounds_3rd[i];
                    t_info.s_info_in = in_sync_info_4th + i;
                    t_info.s_info_out = out_sync_info_4th + i;
                    t4th[i] = new thread(&cv::cuda::HOG::thread_fine_classify, gpu_hog,
                            &(level_nodes[i][3]), fine_init_barrier, t_info);
                    break;
                case fine_CD:
                    t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, costs_1st[i]);
                    t_info.id = task_id++;
                    t_info.phase = PERIOD * g_idx + bound_color_convert + bound_compute_scales;
                    t_info.s_info_in = in_sync_info_1st + i;
                    t_info.s_info_out = out_sync_info_1st + i;
                    t1st[i] = new thread(&cv::cuda::HOG::thread_fine_resize, gpu_hog,
                            &(level_nodes[i][0]), fine_init_barrier, t_info);

                    t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, costs_2nd[i]);
                    t_info.id = task_id++;
                    t_info.phase = t_info.phase + bounds_1st[i];
                    t_info.s_info_in = in_sync_info_2nd + i;
                    t_info.s_info_out = out_sync_info_2nd + i;
                    t2nd[i] = new thread(&cv::cuda::HOG::thread_fine_compute_gradients,
                            gpu_hog, &(level_nodes[i][1]), fine_init_barrier,
                            t_info);

                    t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, costs_3rd[i]);
                    t_info.id = task_id++;
                    t_info.phase = t_info.phase + bounds_2nd[i];
                    t_info.s_info_in = in_sync_info_3rd + i;
                    t_info.s_info_out = out_sync_info_3rd + i;
                    t3rd[i] = new thread(&cv::cuda::HOG::thread_fine_CD,
                            gpu_hog, &(level_nodes[i][2]), fine_init_barrier, t_info);

                    t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, costs_4th[i]);
                    t_info.id = task_id++;
                    t_info.phase = t_info.phase + bounds_3rd[i];
                    t_info.s_info_in = in_sync_info_4th + i;
                    t_info.s_info_out = out_sync_info_4th + i;
                    t4th[i] = new thread(&cv::cuda::HOG::thread_fine_classify, gpu_hog,
                            &(level_nodes[i][3]), fine_init_barrier, t_info);
                    break;
                case fine_DE:
                    t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, costs_1st[i]);
                    t_info.id = task_id++;
                    t_info.phase = PERIOD * g_idx + bound_color_convert + bound_compute_scales;
                    t_info.s_info_in = in_sync_info_1st + i;
                    t_info.s_info_out = out_sync_info_1st + i;
                    t1st[i] = new thread(&cv::cuda::HOG::thread_fine_resize, gpu_hog,
                            &(level_nodes[i][0]), fine_init_barrier, t_info);

                    t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, costs_2nd[i]);
                    t_info.id = task_id++;
                    t_info.phase = t_info.phase + bounds_1st[i];
                    t_info.s_info_in = in_sync_info_2nd + i;
                    t_info.s_info_out = out_sync_info_2nd + i;
                    t2nd[i] = new thread(&cv::cuda::HOG::thread_fine_compute_gradients,
                            gpu_hog, &(level_nodes[i][1]), fine_init_barrier,
                            t_info);

                    t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, costs_3rd[i]);
                    t_info.id = task_id++;
                    t_info.phase = t_info.phase + bounds_2nd[i];
                    t_info.s_info_in = in_sync_info_3rd + i;
                    t_info.s_info_out = out_sync_info_3rd + i;
                    t3rd[i] = new thread(&cv::cuda::HOG::thread_fine_compute_histograms,
                            gpu_hog, &(level_nodes[i][2]), fine_init_barrier, t_info);

                    t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, costs_4th[i]);
                    t_info.id = task_id++;
                    t_info.phase = t_info.phase + bounds_3rd[i];
                    t_info.s_info_in = in_sync_info_4th + i;
                    t_info.s_info_out = out_sync_info_4th + i;
                    t4th[i] = new thread(&cv::cuda::HOG::thread_fine_DE, gpu_hog,
                            &(level_nodes[i][3]), fine_init_barrier, t_info);
                    break;
                case fine_ABC:
                    t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, costs_1st[i]);
                    t_info.id = task_id++;
                    t_info.phase = PERIOD * g_idx + bound_color_convert + bound_compute_scales;
                    t_info.s_info_in = in_sync_info_1st + i;
                    t_info.s_info_out = out_sync_info_1st + i;
                    t1st[i] = new thread(&cv::cuda::HOG::thread_fine_ABC, gpu_hog,
                            &(level_nodes[i][0]), fine_init_barrier, t_info);

                    t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, costs_2nd[i]);
                    t_info.id = task_id++;
                    t_info.phase = t_info.phase + bounds_1st[i];
                    t_info.s_info_in = in_sync_info_2nd + i;
                    t_info.s_info_out = out_sync_info_2nd + i;
                    t2nd[i] = new thread(&cv::cuda::HOG::thread_fine_normalize_histograms,
                            gpu_hog, &(level_nodes[i][1]), fine_init_barrier,
                            t_info);

                    t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, costs_3rd[i]);
                    t_info.id = task_id++;
                    t_info.phase = t_info.phase + bounds_2nd[i];
                    t_info.s_info_in = in_sync_info_3rd + i;
                    t_info.s_info_out = out_sync_info_3rd + i;
                    t3rd[i] = new thread(&cv::cuda::HOG::thread_fine_classify,
                            gpu_hog, &(level_nodes[i][2]), fine_init_barrier, t_info);
                    break;
                case fine_BCD:
                    t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, costs_1st[i]);
                    t_info.id = task_id++;
                    t_info.phase = PERIOD * g_idx + bound_color_convert + bound_compute_scales;
                    t_info.s_info_in = in_sync_info_1st + i;
                    t_info.s_info_out = out_sync_info_1st + i;
                    t1st[i] = new thread(&cv::cuda::HOG::thread_fine_resize, gpu_hog,
                            &(level_nodes[i][0]), fine_init_barrier, t_info);

                    t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, costs_2nd[i]);
                    t_info.id = task_id++;
                    t_info.phase = t_info.phase + bounds_1st[i];
                    t_info.s_info_in = in_sync_info_2nd + i;
                    t_info.s_info_out = out_sync_info_2nd + i;
                    t2nd[i] = new thread(&cv::cuda::HOG::thread_fine_BCD,
                            gpu_hog, &(level_nodes[i][1]), fine_init_barrier,
                            t_info);

                    t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, costs_3rd[i]);
                    t_info.id = task_id++;
                    t_info.phase = t_info.phase + bounds_2nd[i];
                    t_info.s_info_in = in_sync_info_3rd + i;
                    t_info.s_info_out = out_sync_info_3rd + i;
                    t3rd[i] = new thread(&cv::cuda::HOG::thread_fine_classify,
                            gpu_hog, &(level_nodes[i][2]), fine_init_barrier, t_info);
                    break;
                case fine_CDE:
                    t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, costs_1st[i]);
                    t_info.id = task_id++;
                    t_info.phase = PERIOD * g_idx + bound_color_convert + bound_compute_scales;
                    t_info.s_info_in = in_sync_info_1st + i;
                    t_info.s_info_out = out_sync_info_1st + i;
                    t1st[i] = new thread(&cv::cuda::HOG::thread_fine_resize, gpu_hog,
                            &(level_nodes[i][0]), fine_init_barrier, t_info);

                    t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, costs_2nd[i]);
                    t_info.id = task_id++;
                    t_info.phase = t_info.phase + bounds_1st[i];
                    t_info.s_info_in = in_sync_info_2nd + i;
                    t_info.s_info_out = out_sync_info_2nd + i;
                    t2nd[i] = new thread(&cv::cuda::HOG::thread_fine_compute_gradients,
                            gpu_hog, &(level_nodes[i][1]), fine_init_barrier,
                            t_info);

                    t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, costs_3rd[i]);
                    t_info.id = task_id++;
                    t_info.phase = t_info.phase + bounds_2nd[i];
                    t_info.s_info_in = in_sync_info_3rd + i;
                    t_info.s_info_out = out_sync_info_3rd + i;
                    t3rd[i] = new thread(&cv::cuda::HOG::thread_fine_CDE,
                            gpu_hog, &(level_nodes[i][2]), fine_init_barrier, t_info);
                    break;
                case fine_ABCD:
                    t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, costs_1st[i]);
                    t_info.id = task_id++;
                    t_info.phase = PERIOD * g_idx + bound_color_convert + bound_compute_scales;
                    t_info.s_info_in = in_sync_info_1st + i;
                    t_info.s_info_out = out_sync_info_1st + i;
                    t1st[i] = new thread(&cv::cuda::HOG::thread_fine_ABCD, gpu_hog,
                            &(level_nodes[i][0]), fine_init_barrier, t_info);

                    t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, costs_2nd[i]);
                    t_info.id = task_id++;
                    t_info.phase = t_info.phase + bounds_1st[i];
                    t_info.s_info_in = in_sync_info_2nd + i;
                    t_info.s_info_out = out_sync_info_2nd + i;
                    t2nd[i] = new thread(&cv::cuda::HOG::thread_fine_classify,
                            gpu_hog, &(level_nodes[i][1]), fine_init_barrier,
                            t_info);
                    break;
                case fine_BCDE:
                    t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, costs_1st[i]);
                    t_info.id = task_id++;
                    t_info.phase = PERIOD * g_idx + bound_color_convert + bound_compute_scales;
                    t_info.s_info_in = in_sync_info_1st + i;
                    t_info.s_info_out = out_sync_info_1st + i;
                    t1st[i] = new thread(&cv::cuda::HOG::thread_fine_resize, gpu_hog,
                            &(level_nodes[i][0]), fine_init_barrier, t_info);

                    t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, costs_2nd[i]);
                    t_info.id = task_id++;
                    t_info.phase = t_info.phase + bounds_1st[i];
                    t_info.s_info_in = in_sync_info_2nd + i;
                    t_info.s_info_out = out_sync_info_2nd + i;
                    t2nd[i] = new thread(&cv::cuda::HOG::thread_fine_BCDE,
                            gpu_hog, &(level_nodes[i][1]), fine_init_barrier,
                            t_info);
                    break;
                case fine_ABCDE:
                    t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, costs_1st[i]);
                    t_info.id = task_id++;
                    t_info.phase = PERIOD * g_idx + bound_color_convert + bound_compute_scales;
                    t_info.s_info_in = in_sync_info_1st + i;
                    t_info.s_info_out = out_sync_info_1st + i;
                    t1st[i] = new thread(&cv::cuda::HOG::thread_fine_ABCDE, gpu_hog,
                            &(level_nodes[i][0]), fine_init_barrier, t_info);
                    break;
                default:
                    break;
            }
        }

        t_info.relative_deadline = FAIR_LATENESS_PP(m_cpus, t_info.period, cost_collect_locations);
        t_info.id = task_id++;
        // TODO: fix this computation
        t_info.phase = t_info.phase + *std::max_element(bounds_5th, bounds_5th+NUM_SCALE_LEVELS);
        *t7 = new thread(&cv::cuda::HOG::thread_fine_collect_locations, gpu_hog,
                &collect_locations_node, fine_init_barrier, t_info);

        *t8 = new thread(&App::thread_display, this, &display_node,
                fine_init_barrier, g_idx == 0 && args.display);
    }

    /* graph construction finishes */

    printf("Joining pthreads...\n");

    for (int g_idx = 0; g_idx < args.num_fine_graphs; g_idx++) {
        graph_t g = arr_g[g_idx];


        thread* t0 = arr_t0[g_idx];
        thread* t1 = arr_t1[g_idx];
        thread** t1st = arr_t1st + g_idx * NUM_SCALE_LEVELS;
        thread** t2nd = arr_t2nd + g_idx * NUM_SCALE_LEVELS;
        thread** t3rd = arr_t3rd + g_idx * NUM_SCALE_LEVELS;
        thread** t4th = arr_t4th + g_idx * NUM_SCALE_LEVELS;
        thread** t5th = arr_t5th + g_idx * NUM_SCALE_LEVELS;
        thread* t7 = arr_t7[g_idx];
        thread* t8 = arr_t8[g_idx];
        t0->join();
        t1->join();
        delete t0;
        delete t1;
        for (int i=0; i<NUM_SCALE_LEVELS; i++) {
            switch (level_options[i]) {
                case fine_grained:
                    if (t1st[i]->joinable()) t1st[i]->join();
                    if (t2nd[i]->joinable()) t2nd[i]->join();
                    if (t3rd[i]->joinable()) t3rd[i]->join();
                    if (t4th[i]->joinable()) t4th[i]->join();
                    if (t5th[i]->joinable()) t5th[i]->join();
                    break;
                case fine_AB:
                case fine_BC:
                case fine_CD:
                case fine_DE:
                    if (t1st[i]->joinable()) t1st[i]->join();
                    if (t2nd[i]->joinable()) t2nd[i]->join();
                    if (t3rd[i]->joinable()) t3rd[i]->join();
                    if (t4th[i]->joinable()) t4th[i]->join();
                    break;
                case fine_ABC:
                case fine_BCD:
                case fine_CDE:
                    if (t1st[i]->joinable()) t1st[i]->join();
                    if (t2nd[i]->joinable()) t2nd[i]->join();
                    if (t3rd[i]->joinable()) t3rd[i]->join();
                    break;
                case fine_ABCD:
                case fine_BCDE:
                    if (t1st[i]->joinable()) t1st[i]->join();
                    if (t2nd[i]->joinable()) t2nd[i]->join();
                    break;
                case fine_ABCDE:
                    if (t1st[i]->joinable()) t1st[i]->join();
                    break;
                default:
                    break;
            }
            delete t1st[i];
            delete t2nd[i];
            delete t3rd[i];
            delete t4th[i];
            delete t5th[i];
        }
        t7->join();
        t8->join();
        delete t7;
        delete t8;
        CheckError(pgm_destroy_graph(g));
    }
    free(arr_t1);
    free(arr_t1st);
    free(arr_t2nd);
    free(arr_t3rd);
    free(arr_t4th);
    free(arr_t5th);
    free(arr_t7);
    free(arr_t8);

    //CheckError(pgm_destroy_graph(g));
    CheckError(pgm_destroy());
    fprintf(stdout, "cleaned up ...");
}