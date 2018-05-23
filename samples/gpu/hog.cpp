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

pthread_barrier_t init_barrier;
int hog_sample_errors;

__thread char hog_sample_errstr[80];

//#define LOG_DEBUG 1
#define NUM_SCALE_LEVELS 13
#define NUM_FINE_GRAPHS 1
#define CheckError(e) \
do { int __ret = (e); \
if(__ret < 0) { \
    hog_sample_errors++; \
    char* errstr = strerror_r(errno, hog_sample_errstr, sizeof(errstr)); \
    fprintf(stderr, "%lu: Error %d (%s (%d)) @ %s:%s:%d\n",  \
            pthread_self(), __ret, errstr, errno, __FILE__, __FUNCTION__, __LINE__); \
}}while(0)

/* Next, we define period and execution cost to be constant.
 * These are only constants for convenience in this example, they can be
 * determined at run time, e.g., from command line parameters.
 *
 * These are in milliseconds.
 */
#define PERIOD            50
#define RELATIVE_DEADLINE 50
#define EXEC_COST         5

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

enum scheduling_option
{
    end_to_end = 0,
    coarse_grained,
    fine_grained,
    coarse_unrolled,
    scheduling_option_end,
};

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
};


struct task_info
{
    int id;
    /* real-time parameters in milliseconds */
    int period;
    int phase;
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
    void sched_coarse_grained_hog(cv::Ptr<cv::cuda::HOG> gpu_hog, cv::HOGDescriptor cpu_hog, Mat* frames);
    void sched_coarse_grained_unrolled_for_hog(cv::Ptr<cv::cuda::HOG> gpu_hog, cv::HOGDescriptor cpu_hog, Mat* frames);
    void sched_fine_grained_hog(cv::Ptr<cv::cuda::HOG> gpu_hog, cv::HOGDescriptor cpu_hog, Mat* frames);
    void thread_color_convert(node_t *_node, pthread_barrier_t* init_barrier,
            cv::Ptr<cv::cuda::HOG> gpu_hog, cv::HOGDescriptor cpu_hog, Mat* frames);

    void* thread_display(node_t* node, pthread_barrier_t* init_barrier);

    void handleKey(char key);

    void hogWorkBegin();
    void hogWorkEnd();
    string hogWorkFps() const;

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
    display = false;
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

void* App::thread_display(node_t* _node, pthread_barrier_t* init_barrier)
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


                printf("%d response time: %f\n", in_buf->frame_index, (hog_work_end - in_buf->start_time) / getTickFrequency());

                // Draw positive classified windows
                if (args.display) {
                    for (size_t i = 0; i < in_buf->found->size(); i++) {
                        Rect r = (*in_buf->found)[i];
                        rectangle(*in_buf->img_to_show, r.tl(), r.br(), Scalar(0, 255, 0), 3);
#ifdef LOG_DEBUG
                        fprintf(stdout, "point: %d, %d, %d, %d\n", r.tl().x, r.tl().y, r.br().x, r.br().y);
#endif
                    }

                    if (use_gpu)
                        putText(*in_buf->img_to_show, "Mode: GPU", Point(5, 25), FONT_HERSHEY_SIMPLEX, 1., Scalar(255, 100, 0), 2);
                    else
                        putText(*in_buf->img_to_show, "Mode: CPU", Point(5, 25), FONT_HERSHEY_SIMPLEX, 1., Scalar(255, 100, 0), 2);
                    putText(*in_buf->img_to_show, "FPS HOG: " + hogWorkFps(), Point(5, 65), FONT_HERSHEY_SIMPLEX, 1., Scalar(255, 100, 0), 2);
                    putText(*in_buf->img_to_show, "FPS total: " + workFps(), Point(5, 105), FONT_HERSHEY_SIMPLEX, 1., Scalar(255, 100, 0), 2);
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

void App::sched_coarse_grained_hog(cv::Ptr<cv::cuda::HOG> gpu_hog, cv::HOGDescriptor cpu_hog, Mat* frames)
{
    /* graph construction */
    graph_t g;
    node_t color_convert_node;
    node_t vxHOGCells_node;
    node_t vxHOGFeatures_node;
    node_t classify_node;
    node_t collect_locations_node;
    node_t display_node;
    edge_t e0_1, e1_2, e2_3, e3_4, e4_5;

    CheckError(pgm_init("/tmp/graphs", 1));
    CheckError(pgm_init_graph(&g, "hog"));

    CheckError(pgm_init_node(&color_convert_node, g, "color_convert"));
    CheckError(pgm_init_node(&vxHOGCells_node, g, "detect"));
    CheckError(pgm_init_node(&vxHOGFeatures_node, g, "detect"));
    CheckError(pgm_init_node(&classify_node, g, "classify"));
    CheckError(pgm_init_node(&collect_locations_node, g, "classify"));
    CheckError(pgm_init_node(&display_node, g, "display"));

    edge_attr_t fast_mq_attr;
    memset(&fast_mq_attr, 0, sizeof(fast_mq_attr));
    fast_mq_attr.mq_maxmsg = 1; /* root required for higher values */
    fast_mq_attr.type = pgm_fast_mq_edge;

    fast_mq_attr.nr_produce = sizeof(struct params_compute);
    fast_mq_attr.nr_consume = sizeof(struct params_compute);
    fast_mq_attr.nr_threshold = sizeof(struct params_compute);
    CheckError(pgm_init_edge(&e0_1, color_convert_node, vxHOGCells_node, "e0_1", &fast_mq_attr));

    fast_mq_attr.nr_produce = sizeof(struct params_normalize);
    fast_mq_attr.nr_consume = sizeof(struct params_normalize);
    fast_mq_attr.nr_threshold = sizeof(struct params_normalize);
    CheckError(pgm_init_edge(&e1_2, vxHOGCells_node, vxHOGFeatures_node, "e1_2", &fast_mq_attr));

    fast_mq_attr.nr_produce = sizeof(struct params_classify);
    fast_mq_attr.nr_consume = sizeof(struct params_classify);
    fast_mq_attr.nr_threshold = sizeof(struct params_classify);
    CheckError(pgm_init_edge(&e2_3, vxHOGFeatures_node, classify_node, "e2_3", &fast_mq_attr));

    fast_mq_attr.nr_produce = sizeof(struct params_collect_locations);
    fast_mq_attr.nr_consume = sizeof(struct params_collect_locations);
    fast_mq_attr.nr_threshold = sizeof(struct params_collect_locations);
    CheckError(pgm_init_edge(&e3_4, classify_node, collect_locations_node, "e3_4", &fast_mq_attr));

    fast_mq_attr.nr_produce = sizeof(struct params_display);
    fast_mq_attr.nr_consume = sizeof(struct params_display);
    fast_mq_attr.nr_threshold = sizeof(struct params_display);
    CheckError(pgm_init_edge(&e4_5, collect_locations_node, display_node, "e4_5", &fast_mq_attr));

    pthread_barrier_init(&init_barrier, 0, 6);

    thread t1(&cv::cuda::HOG::thread_vxHOGCells, gpu_hog, &vxHOGCells_node, &init_barrier, 0);
    thread t2(&cv::cuda::HOG::thread_vxHOGFeatures, gpu_hog, &vxHOGFeatures_node, &init_barrier, 0);
    thread t3(&cv::cuda::HOG::thread_classify, gpu_hog, &classify_node, &init_barrier, 0);
    thread t4(&cv::cuda::HOG::thread_collect_locations, gpu_hog, &collect_locations_node, &init_barrier, 0);
    thread t5(&App::thread_display, this, &display_node, &init_barrier);

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

    pthread_barrier_wait(&init_barrier);

    int count_frame = 0;
    while (count_frame < args.count && running) {

        for (int j=0; j<100; j++) {
            if (count_frame >= args.count)
                break;
            frame = frames[j];
            usleep(10000);
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
        }
    }

    free(out_edge);
    CheckError(pgm_terminate(color_convert_node));
    pthread_barrier_wait(&init_barrier);
    CheckError(pgm_release_node(color_convert_node));
    printf("Joining pthreads...\n");
    t1.join();
    t2.join();
    t3.join();
    t4.join();
    t5.join();
    //t6.join();
    //t7.join();
    //t8.join();

    CheckError(pgm_destroy_graph(g));
    CheckError(pgm_destroy());
}

void App::sched_coarse_grained_unrolled_for_hog(cv::Ptr<cv::cuda::HOG> gpu_hog, cv::HOGDescriptor cpu_hog, Mat* frames)
{

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

    CheckError(pgm_init("/tmp/graphs", 1));
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
    fast_mq_attr.mq_maxmsg = 1; /* root required for higher values */
    fast_mq_attr.type = pgm_fast_mq_edge;

    fast_mq_attr.nr_produce = sizeof(struct params_compute);
    fast_mq_attr.nr_consume = sizeof(struct params_compute);
    fast_mq_attr.nr_threshold = sizeof(struct params_compute);
    CheckError(pgm_init_edge(&e0_1, color_convert_node, compute_scales_node, "e0_1", &fast_mq_attr));

    char buf[30];
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

    pthread_barrier_init(&init_barrier, 0, 3 * NUM_SCALE_LEVELS + 4);

    thread** t2 = (thread**) calloc(NUM_SCALE_LEVELS, sizeof(std::thread *));
    thread** t3 = (thread**) calloc(NUM_SCALE_LEVELS, sizeof(std::thread *));
    thread** t4 = (thread**) calloc(NUM_SCALE_LEVELS, sizeof(std::thread *));

    thread t1(&cv::cuda::HOG::thread_fine_compute_scales, gpu_hog, &compute_scales_node, &init_barrier, 5);
    for (int i=0; i<NUM_SCALE_LEVELS; i++) {
        t2[i] = new thread(&cv::cuda::HOG::thread_unrolled_vxHOGCells, gpu_hog, &unrolled_vxHOGCells_node[i], &init_barrier, 9);
        t3[i] = new thread(&cv::cuda::HOG::thread_fine_normalize_histograms, gpu_hog, &normalize_node[i], &init_barrier, 20);
        t4[i] = new thread(&cv::cuda::HOG::thread_fine_classify, gpu_hog, &classify_node[i], &init_barrier, 22);
    }
    thread t5(&cv::cuda::HOG::thread_fine_collect_locations, gpu_hog, &collect_locations_node, &init_barrier, 24);
    thread t6(&App::thread_display, this, &display_node, &init_barrier);

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

    struct rt_task param;
    init_rt_task_param(&param);
    param.exec_cost = ms2ns(EXEC_COST);
    param.period = ms2ns(PERIOD);
    param.relative_deadline = ms2ns(RELATIVE_DEADLINE);
    param.budget_policy = NO_ENFORCEMENT;
    param.cls = RT_CLASS_SOFT;
    param.priority = LITMUS_LOWEST_PRIORITY;
    CALL( init_litmus() );
    CALL( set_rt_task_param(gettid(), &param) );
    CALL( task_mode(LITMUS_RT_TASK) );
    CALL( wait_for_ts_release() );

    int count_frame = 0;
    while (count_frame < args.count && running) {

        for (int j=0; j<100; j++) {
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
            sleep_next_period();
        }
    }
    /*****
     * 6) Transition to background mode.
     */
    CALL( task_mode(BACKGROUND_TASK) );

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
}

void App::thread_color_convert(node_t *_node, pthread_barrier_t* init_barrier,
        cv::Ptr<cv::cuda::HOG> gpu_hog, cv::HOGDescriptor cpu_hog, Mat* frames)
{
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

    struct rt_task param;
    init_rt_task_param(&param);
    param.exec_cost = ms2ns(EXEC_COST);
    param.period = ms2ns(PERIOD);
    param.relative_deadline = ms2ns(RELATIVE_DEADLINE);
    param.budget_policy = NO_ENFORCEMENT;
    param.cls = RT_CLASS_SOFT;
    param.priority = LITMUS_LOWEST_PRIORITY;
    CALL( init_litmus() );
    CALL( set_rt_task_param(gettid(), &param) );
    CALL( task_mode(LITMUS_RT_TASK) );
    CALL( wait_for_ts_release() );

    int count_frame = 0;
    while (count_frame < args.count && running) {
        for (int j=0; j<100; j++) {
            if (count_frame >= args.count)
                break;
            frame = frames[j];
            //fprintf(stdout, "0 fires: image_to_show: %p, found: %p\n", img, found);
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
                out_buf->frame_index = count_frame;
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
            sleep_next_period();
        }
    }
    CALL( task_mode(BACKGROUND_TASK) );

    free(out_edge);
    CheckError(pgm_terminate(node));

    pthread_barrier_wait(init_barrier);

    CheckError(pgm_release_node(node));
}

void App::sched_fine_grained_hog(cv::Ptr<cv::cuda::HOG> gpu_hog, cv::HOGDescriptor cpu_hog, Mat* frames)
{
    pthread_barrier_t arr_fine_init_barrier[NUM_FINE_GRAPHS];
    /* graph construction */
    graph_t arr_g                       [NUM_FINE_GRAPHS];
    node_t arr_color_convert_node       [NUM_FINE_GRAPHS];
    node_t arr_compute_scales_node      [NUM_FINE_GRAPHS];
    node_t arr_resize_node              [NUM_FINE_GRAPHS][NUM_SCALE_LEVELS];
    node_t arr_compute_gradients_node   [NUM_FINE_GRAPHS][NUM_SCALE_LEVELS];
    node_t arr_compute_histograms_node  [NUM_FINE_GRAPHS][NUM_SCALE_LEVELS];
    node_t arr_normalize_node           [NUM_FINE_GRAPHS][NUM_SCALE_LEVELS];
    node_t arr_classify_node            [NUM_FINE_GRAPHS][NUM_SCALE_LEVELS];
    node_t arr_collect_locations_node   [NUM_FINE_GRAPHS];
    node_t arr_display_node             [NUM_FINE_GRAPHS];
    edge_t arr_e0_1[NUM_FINE_GRAPHS];
    edge_t arr_e1_2[NUM_FINE_GRAPHS][NUM_SCALE_LEVELS];
    edge_t arr_e2_3[NUM_FINE_GRAPHS][NUM_SCALE_LEVELS];
    edge_t arr_e3_4[NUM_FINE_GRAPHS][NUM_SCALE_LEVELS];
    edge_t arr_e4_5[NUM_FINE_GRAPHS][NUM_SCALE_LEVELS];
    edge_t arr_e5_6[NUM_FINE_GRAPHS][NUM_SCALE_LEVELS];
    edge_t arr_e6_7[NUM_FINE_GRAPHS][NUM_SCALE_LEVELS];
    edge_t arr_e7_8[NUM_FINE_GRAPHS];

    thread** arr_t0 = (thread**) calloc(NUM_FINE_GRAPHS, sizeof(std::thread *));
    thread** arr_t1 = (thread**) calloc(NUM_FINE_GRAPHS, sizeof(std::thread *));
    thread** arr_t2 = (thread**) calloc(NUM_FINE_GRAPHS * NUM_SCALE_LEVELS, sizeof(std::thread *));
    thread** arr_t3 = (thread**) calloc(NUM_FINE_GRAPHS * NUM_SCALE_LEVELS, sizeof(std::thread *));
    thread** arr_t4 = (thread**) calloc(NUM_FINE_GRAPHS * NUM_SCALE_LEVELS, sizeof(std::thread *));
    thread** arr_t5 = (thread**) calloc(NUM_FINE_GRAPHS * NUM_SCALE_LEVELS, sizeof(std::thread *));
    thread** arr_t6 = (thread**) calloc(NUM_FINE_GRAPHS * NUM_SCALE_LEVELS, sizeof(std::thread *));
    thread** arr_t7 = (thread**) calloc(NUM_FINE_GRAPHS, sizeof(std::thread *));
    thread** arr_t8 = (thread**) calloc(NUM_FINE_GRAPHS, sizeof(std::thread *));

    CheckError(pgm_init("/tmp/graphs", 1));

    for (int g_idx = 0; g_idx < NUM_FINE_GRAPHS; g_idx++) {
        pthread_barrier_t* fine_init_barrier = arr_fine_init_barrier + g_idx;
        graph_t* g_ptr = arr_g + g_idx;
        node_t* color_convert_node      = arr_color_convert_node + g_idx;
        node_t* compute_scales_node     = arr_compute_scales_node + g_idx;
        node_t* resize_node             = arr_resize_node[g_idx];
        node_t* compute_gradients_node  = arr_compute_gradients_node[g_idx];
        node_t* compute_histograms_node = arr_compute_histograms_node[g_idx];
        node_t* normalize_node          = arr_normalize_node[g_idx];
        node_t* classify_node           = arr_classify_node[g_idx];
        node_t* collect_locations_node  = arr_collect_locations_node + g_idx;
        node_t* display_node            = arr_display_node + g_idx;
        edge_t* e0_1 = arr_e0_1 + g_idx;
        edge_t* e1_2 = arr_e1_2[g_idx];
        edge_t* e2_3 = arr_e2_3[g_idx];
        edge_t* e3_4 = arr_e3_4[g_idx];
        edge_t* e4_5 = arr_e4_5[g_idx];
        edge_t* e5_6 = arr_e5_6[g_idx];
        edge_t* e6_7 = arr_e6_7[g_idx];
        edge_t* e7_8 = arr_e7_8 + g_idx;

        thread** t1 = arr_t1 + g_idx;
        thread** t2 = arr_t2 + g_idx * NUM_SCALE_LEVELS;
        thread** t3 = arr_t3 + g_idx * NUM_SCALE_LEVELS;
        thread** t4 = arr_t4 + g_idx * NUM_SCALE_LEVELS;
        thread** t5 = arr_t5 + g_idx * NUM_SCALE_LEVELS;
        thread** t6 = arr_t6 + g_idx * NUM_SCALE_LEVELS;
        thread** t7 = arr_t7 + g_idx;
        thread** t8 = arr_t8 + g_idx;

        char buf[30];
        sprintf(buf, "hog_%d", g_idx);
        CheckError(pgm_init_graph(g_ptr, buf));
        graph_t g = *g_ptr;

        CheckError(pgm_init_node(color_convert_node, g, "color_convert"));
        CheckError(pgm_init_node(compute_scales_node, g, "compute_scales"));
        for (int i=0; i<NUM_SCALE_LEVELS; i++) {
            CheckError(pgm_init_node(resize_node + i, g, "resize"));
            CheckError(pgm_init_node(compute_gradients_node + i, g, "compute_gradients"));
            CheckError(pgm_init_node(compute_histograms_node + i, g, "compute_histograms"));
            CheckError(pgm_init_node(normalize_node + i, g, "normalize"));
            CheckError(pgm_init_node(classify_node + i, g, "classify"));
        }
        CheckError(pgm_init_node(collect_locations_node, g, "collect_locations"));
        CheckError(pgm_init_node(display_node, g, "display"));

        edge_attr_t fast_mq_attr;
        memset(&fast_mq_attr, 0, sizeof(fast_mq_attr));
        fast_mq_attr.mq_maxmsg = 20; /* root required for values larger than 10 */
        fast_mq_attr.type = pgm_fast_mq_edge;

        fast_mq_attr.nr_produce = sizeof(struct params_compute);
        fast_mq_attr.nr_consume = sizeof(struct params_compute);
        fast_mq_attr.nr_threshold = sizeof(struct params_compute);
        CheckError(pgm_init_edge(e0_1, *color_convert_node, *compute_scales_node, "e0_1", &fast_mq_attr));

        for (int i=0; i<NUM_SCALE_LEVELS; i++) {
            sprintf(buf, "e1_2_%d", i);
            fast_mq_attr.nr_produce = sizeof(struct params_resize);
            fast_mq_attr.nr_consume = sizeof(struct params_resize);
            fast_mq_attr.nr_threshold = sizeof(struct params_resize);
            CheckError(pgm_init_edge(e1_2 + i,
                        *compute_scales_node,
                        resize_node[i], buf, &fast_mq_attr));

            sprintf(buf, "e2_3_%d", i);
            fast_mq_attr.nr_produce = sizeof(struct params_compute_gradients);
            fast_mq_attr.nr_consume = sizeof(struct params_compute_gradients);
            fast_mq_attr.nr_threshold = sizeof(struct params_compute_gradients);
            CheckError(pgm_init_edge(e2_3 + i,
                        resize_node[i],
                        compute_gradients_node[i], buf,
                        &fast_mq_attr));

            sprintf(buf, "e3_4_%d", i);
            fast_mq_attr.nr_produce = sizeof(struct params_compute_histograms);
            fast_mq_attr.nr_consume = sizeof(struct params_compute_histograms);
            fast_mq_attr.nr_threshold = sizeof(struct params_compute_histograms);
            CheckError(pgm_init_edge(e3_4 + i,
                        compute_gradients_node[i],
                        compute_histograms_node[i], buf,
                        &fast_mq_attr));

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
                        *collect_locations_node, buf,
                        &fast_mq_attr));
        }

        fast_mq_attr.nr_produce = sizeof(struct params_display);
        fast_mq_attr.nr_consume = sizeof(struct params_display);
        fast_mq_attr.nr_threshold = sizeof(struct params_display);
        CheckError(pgm_init_edge(e7_8, *collect_locations_node, *display_node, "e7_8", &fast_mq_attr));

        pthread_barrier_init(fine_init_barrier, 0, 5 * NUM_SCALE_LEVELS + 4);

        *t1 = new thread(&cv::cuda::HOG::thread_fine_compute_scales, gpu_hog, compute_scales_node, fine_init_barrier, 5);
        for (int i=0; i<NUM_SCALE_LEVELS; i++) {
            t2[i] = new thread(&cv::cuda::HOG::thread_fine_resize, gpu_hog, resize_node + i, fine_init_barrier, 6);
            t3[i] = new thread(&cv::cuda::HOG::thread_fine_compute_gradients, gpu_hog, compute_gradients_node + i, fine_init_barrier, 8);
            t4[i] = new thread(&cv::cuda::HOG::thread_fine_compute_histograms, gpu_hog, compute_histograms_node + i, fine_init_barrier, 13);
            t5[i] = new thread(&cv::cuda::HOG::thread_fine_normalize_histograms, gpu_hog, normalize_node + i, fine_init_barrier, 20);
            t6[i] = new thread(&cv::cuda::HOG::thread_fine_classify, gpu_hog, classify_node + i, fine_init_barrier, 22);
        }
        *t7 = new thread(&cv::cuda::HOG::thread_fine_collect_locations, gpu_hog, collect_locations_node, fine_init_barrier, 24);
        *t8 = new thread(&App::thread_display, this, display_node, fine_init_barrier);
    }

    /* graph construction finishes */

    for (int g_idx = 0; g_idx < NUM_FINE_GRAPHS; g_idx++) {
        thread** t0 = arr_t0 + g_idx;
        *t0 = new thread(&App::thread_color_convert, this,
                arr_color_convert_node + g_idx, arr_fine_init_barrier + g_idx,
                gpu_hog, cpu_hog, frames);
    }

    printf("Joining pthreads...\n");

    for (int g_idx = 0; g_idx < NUM_FINE_GRAPHS; g_idx++) {
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
	struct rt_task param;
    unsigned int job_no = 0;

	/* Setup task parameters */
	init_rt_task_param(&param);
	param.exec_cost = ms2ns(EXEC_COST);
	param.period = ms2ns(PERIOD);
	param.relative_deadline = ms2ns(RELATIVE_DEADLINE);

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

    /*****
     * 4) Transition to real-time mode.
     */
    CALL( task_mode(LITMUS_RT_TASK) );
    CALL( wait_for_ts_release() );

    /* The task is now executing as a real-time task if the call didn't fail.
     */
    while (count_frame < args.count && running) {
        for (int j=0; j<100; j++) {
            get_job_no(&job_no);
            fprintf(stdout, "job %d\n", job_no);
            if (count_frame >= args.count)
                break;
            frame = frames[j];
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
            sleep_next_period();
        }
    }
    /*****
     * 6) Transition to background mode.
     */
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
            //sched_etoe_hog(gpu_hog, cpu_hog);
            sched_etoe_hog_preload(gpu_hog, cpu_hog, frames);
            break;
        case coarse_grained:
            sched_coarse_grained_hog(gpu_hog, cpu_hog, frames);
            break;
        case fine_grained:
            sched_fine_grained_hog(gpu_hog, cpu_hog, frames);
            break;
        case coarse_unrolled:
            sched_coarse_grained_unrolled_for_hog(gpu_hog, cpu_hog, frames);
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
