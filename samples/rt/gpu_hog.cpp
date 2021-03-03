#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sstream>
#include <thread>
#include <opencv2/core/utility.hpp>
#include "opencv2/rtcudaobjdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/imgproc.hpp"

#include <unistd.h>
#include <errno.h>
#include <pthread.h>
#include <signal.h>
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

#define SAMPLE_START_LOCK(s, t) \
            gpu_hog->lock_fzlp(omlp_sem_od); \
            gpu_hog->wait_forbidden_zone(omlp_sem_od, t); \
            s = litmus_clock();

#define SAMPLE_STOP_LOCK(l, t) \
                l = litmus_clock() - fz_start; \
                fprintf(stdout, "[%d | %d] Computation %d took %llu microseconds.\n", \
                        gettid(), getpid(), t, fz_len / 1000); \
                gpu_hog->unlock_fzlp(omlp_sem_od);

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
    fprintf(stderr, "%d: Error %d (%s (%d)) @ %s:%s:%d\n",  \
            gettid(), __ret, errstr, errno, __FILE__, __FUNCTION__, __LINE__); \
}}while(0)


/* Catch errors.
 */
#define CALL( exp ) do { \
    int _ret; \
    _ret = exp; \
    if (_ret != 0) \
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
    int num_hog_inst;
    int num_fine_graphs;
    int cluster;
    int task_id;
    bool realtime;
    bool early;

    string config_filepath;
    std::vector< std::vector<node_config> > level_configurations;
    bool merge_color_convert;
    std::vector<node_config> source_configuration;
    std::vector<node_config> sink_configuration;
    std::vector<float> non_level_costs;
    std::vector< std::vector<float> > level_costs;
    std::vector<float> non_level_bounds;
    std::vector< std::vector<float> > level_bounds;

private:
    void parseLevelConfig(std::string line, std::vector<std::vector<node_config>> &config);
    void parseSourceSinkConfig(std::string line, std::vector<node_config> &config);

    void parseNonLevelCosts(std::string line, std::vector<float> &costs);
    void parseLevelCosts(std::string line, std::vector<std::vector<float>> &costs);

    void parseGraphConfiguration(char *filepath);
};

struct params_color_convert
{
    Mat * frame;
    Mat * img;
    cv::cuda::GpuMat * gpu_img;
    cv::cuda::GpuMat ** grad_array;
    cv::cuda::GpuMat ** qangle_array;
    cv::cuda::GpuMat ** block_hists_array;
    cv::cuda::GpuMat ** smaller_img_array;
    cv::cuda::GpuMat ** labels_array;
    std::vector<Rect> * found;
    Mat * img_to_show;
    size_t frame_index;
    lt_t start_time;
    lt_t end_time;
};

struct params_compute  // a.k.a. compute scales node
{
    cv::cuda::GpuMat * gpu_img;
    cv::cuda::GpuMat ** grad_array;
    cv::cuda::GpuMat ** qangle_array;
    cv::cuda::GpuMat ** block_hists_array;
    cv::cuda::GpuMat ** smaller_img_array;
    cv::cuda::GpuMat ** labels_array;
    std::vector<Rect> * found;
    Mat * img_to_show;
    size_t frame_index;
    lt_t start_time;
    lt_t end_time;
};

/* fine-grained */
struct params_resize
{
    cv::cuda::GpuMat * gpu_img;
    cv::cuda::GpuMat * grad;
    cv::cuda::GpuMat * qangle;
    cv::cuda::GpuMat * block_hists;
    std::vector<Rect> * found;
    Mat * img_to_show;
    cv::cuda::GpuMat * smaller_img;
    cv::cuda::GpuMat * labels;
    std::vector<double> * level_scale;
    std::vector<double> * confidences;
    int index;
    size_t frame_index;
    lt_t start_time;
    lt_t end_time;
};

struct params_compute_gradients
{
    cv::cuda::GpuMat * gpu_img;
    cv::cuda::GpuMat * grad;
    cv::cuda::GpuMat * qangle;
    cv::cuda::GpuMat * block_hists;
    std::vector<Rect> * found;
    Mat * img_to_show;
    cv::cuda::GpuMat * smaller_img;
    cv::cuda::GpuMat * labels;
    std::vector<double> * level_scale;
    std::vector<double> * confidences;
    int index;
    size_t frame_index;
    lt_t start_time;
    lt_t end_time;
};

struct params_compute_histograms
{
    cv::cuda::GpuMat * gpu_img;
    cv::cuda::GpuMat * block_hists;
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
    lt_t start_time;
    lt_t end_time;
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
    lt_t start_time;
    lt_t end_time;
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
    lt_t start_time;
    lt_t end_time;
};

struct params_fine_collect_locations
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
    lt_t start_time;
    lt_t end_time;
};

struct params_display
{
    std::vector<Rect> * found;
    Mat * img_to_show;
    size_t frame_index;
    lt_t start_time;
    lt_t end_time;
};


class App
{
public:
    App(const Args& s);
    void run();

    void sched_configurable_hog(cv::Ptr<cv::cuda::HOG_RT> gpu_hog, cv::HOGDescriptor cpu_hog, Mat* frames);

    void thread_fine_CC_S_ABCDE(node_t node, pthread_barrier_t* init_barrier,
                                cv::Ptr<cv::cuda::HOG_RT> gpu_hog,
                                Mat* frames, struct task_info t_info, int graph_idx);

    void thread_image_acquisition(node_t node, pthread_barrier_t* init_barrier,
                                  cv::Ptr<cv::cuda::HOG_RT> gpu_hog, cv::HOGDescriptor cpu_hog,
                                  Mat* frames, struct task_info t_info, int graph_idx);
    void* thread_display(node_t node, pthread_barrier_t* init_barrier, bool shouldDisplay);

    void thread_color_convert(node_t node, pthread_barrier_t* init_barrier,
                              cv::Ptr<cv::cuda::HOG_RT> gpu_hog,
                              struct task_info t_info);


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
         << "  [--sched <int>] # scheduling option (0:end-to-end, 1:coarse-grained, 2:fine-grained, 3:unrolled-coarse-grained, 4:configurable-nodes)\n"
         << "  [--count <int>] # num of frames to process\n"
         << "  [--graph_bound <int>] # response time bound of fine-grained HOG\n"
         << "  [--cluster <int>] # cluster ID of this task\n"
         << "  [--id <int>] # task ID of this task\n"
         << "  [--rt <true/false>] # whether to run under the LITMUS^RT scheduler\n"
         << "  [--level_config_file <file_path>] # file specifying the graph configuration for configurable-nodes scheduling (defaults to fine-grained)\n"
         << "  [--display <true/false>] # whether to display the resulting frame\n";

    help_showed = true;
}

int main(int argc, char** argv)
{
    struct sigaction handler;
    memset(&handler, 0, sizeof(handler));
    handler.sa_handler = cv::cuda::HOG_RT::default_fz_sig_hndlr;
    sigaction(SIGSYS, &handler, NULL);
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

    sched = CONFIGURABLE;
    count = 1000;
    num_hog_inst = 1;
    num_fine_graphs = 1;
    display = false;
    cluster = -1;
    task_id = 0;
    realtime = true;
    early = true;

    //  Default to fine-grained levels with all nodes separate
    for (unsigned i = 0; i < NUM_SCALE_LEVELS; i++)
    {
        std::vector<node_config> level_config = { NODE_A, NODE_B, NODE_C, NODE_D, NODE_E };
        level_configurations.push_back(level_config);

        std::vector<float> costs = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
        level_costs.push_back(costs);

        std::vector<float> bounds = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
        level_bounds.push_back(bounds);
    }

    // Handle the other nodes
    for (unsigned i = 0; i < 3; i++)
    {
        non_level_costs.push_back(0.0f);

        non_level_bounds.push_back(0.0f);
    }

    // Default to not merging color convert in with the source
    merge_color_convert = false;

    // Default to not merging with the source ("compute scale levels")
    for (unsigned  i = 0; i < NUM_SCALE_LEVELS; i++)
    {
        source_configuration.push_back(NODE_NONE);
    }

    // Default to not merging with the sink ("collect locations")
    for (unsigned  i = 0; i < NUM_SCALE_LEVELS; i++)
    {
        sink_configuration.push_back(NODE_NONE);
    }
}

void Args::parseLevelConfig(std::string line, std::vector<std::vector<node_config>> &config)
{
    std::vector<node_config> level_config;

    size_t prev_pos = 0;
    size_t pos = 0;
    do
    {
        pos = line.find(" ", prev_pos);

        std::string node_config_str = line.substr(prev_pos, pos - prev_pos);
        node_config node = (node_config) stoi(node_config_str);

        // If the node is valid, add it (it might be NODE_NONE if the entire
        // level is merged into the node before or after the split)
        if (node != NODE_NONE)
        {
            level_config.push_back(node);
        }

        prev_pos = pos + 1;
    } while (pos != string::npos);

    config.push_back(level_config);
}

void Args::parseSourceSinkConfig(std::string line, std::vector<node_config> &config)
{
    size_t pos = line.find(" ", 0);

    std::string level_str = line.substr(0, pos);
    int level = stoi(level_str);

    std::string node_config_str = line.substr(pos+1);
    node_config node = (node_config) stoi(node_config_str);

    config[level] = node;
}

void Args::parseNonLevelCosts(std::string line, std::vector<float> &costs)
{
    size_t pos = line.find(" ", 0);

    std::string node_cost_str = line.substr(0, pos);
    float val = stof(node_cost_str);

    costs.push_back(val);
}

void Args::parseLevelCosts(std::string line, std::vector<std::vector<float>> &costs)
{
    std::vector<float> level_cost;

    size_t prev_pos = 0;
    size_t pos = 0;
    do
    {
        pos = line.find(" ", prev_pos);

        if (pos > prev_pos + 1)
        {
            std::string node_cost_str = line.substr(prev_pos, pos - prev_pos);
            float val = stof(node_cost_str);

            level_cost.push_back(val);
        }

        prev_pos = pos + 1;
    } while (pos != string::npos);

    costs.push_back(level_cost);
}

void Args::parseGraphConfiguration(char *filepath)
{
    std::ifstream infile(filepath);

    std::vector< std::vector<node_config> > config;

    std::vector<float> other_costs;
    std::vector< std::vector<float> > costs;
    std::vector<float> other_bounds;
    std::vector< std::vector<float> > bounds;

    std::vector<node_config> source_config(NUM_SCALE_LEVELS, NODE_NONE);
    std::vector<node_config> sink_config(NUM_SCALE_LEVELS, NODE_NONE);

    // Loop until the end of the file is reached
    std::string line;
    while (std::getline(infile, line))
    {
        // Get the first token in the line
        size_t pos = line.find(" ", 0);
        std::string token = line.substr(0, pos);

        if (token == "L")
        {
            this->parseLevelConfig(line.substr(pos+1), config);
        }
        else if (token == "CC")
        {
            this->merge_color_convert = true;
        }
        else if (token == "S")
        {
            this->parseSourceSinkConfig(line.substr(pos+1), source_config);
        }
        else if (token == "T")
        {
            this->parseSourceSinkConfig(line.substr(pos+1), sink_config);
        }
        else if (token == "C")
        {
            this->parseNonLevelCosts(line.substr(pos+1), other_costs);
        }
        else if (token == "CL")
        {
            this->parseLevelCosts(line.substr(pos+1), costs);
        }
        else if (token == "B")
        {
            this->parseNonLevelCosts(line.substr(pos+1), other_bounds);
        }
        else if (token == "BL")
        {
            this->parseLevelCosts(line.substr(pos+1), bounds);
        }
    }

    this->level_configurations = config;

    this->source_configuration = source_config;
    this->sink_configuration = sink_config;

    this->non_level_costs = other_costs;
    this->level_costs = costs;
    this->non_level_bounds = other_bounds;
    this->level_bounds = bounds;
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
            if (sched < 0 || sched >= SCHEDULING_OPTION_END)
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
        else if (string(argv[i]) == "--num_inst") {
            int num_inst = atoi(argv[++i]);
            if (num_inst <= 0)
                throw runtime_error((string("non-positive number of HOG instances: ") + argv[i]));
            args.num_hog_inst = num_inst;
        }
        else if (string(argv[i]) == "--cluster") { args.cluster = atoi(argv[++i]); }
        else if (string(argv[i]) == "--id") { args.task_id = atoi(argv[++i]); }
        else if (string(argv[i]) == "--rt") args.realtime = (string(argv[++i]) == "true");
        else if (string(argv[i]) == "--early") args.early = (string(argv[++i]) == "true");
        else if (string(argv[i]) == "--display") args.display = (string(argv[++i]) == "true");
        else if (string(argv[i]) == "--level_config_file") { args.parseGraphConfiguration(argv[++i]); }
        else if (args.src.empty()) args.src = argv[i];
        else throw runtime_error((string("unknown key: ") + argv[i]));
    }
    return args;
}


App::App(const Args& s)
{
    CALL(cudaSetDeviceFlags(2));
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

void* App::thread_display(node_t node, pthread_barrier_t* init_barrier, bool shouldDisplay)
{
    fprintf(stdout, "node name: display\n");
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
        fprintf(stderr, "display in buffer is NULL\n");

    struct sigaction handler;
    memset(&handler, 0, sizeof(handler));
    handler.sa_handler = cv::cuda::HOG_RT::default_fz_sig_hndlr;
    sigaction(SIGSYS, &handler, NULL);

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


                // printf("%lu response time: %f\n", in_buf->frame_index, (hog_work_end - in_buf->start_time) / getTickFrequency());

                // Draw positive classified windows
                if (shouldDisplay || args.write_video)
                {
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
                }

                if (shouldDisplay)
                {
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

    /* end is finished */
    pthread_barrier_wait(init_barrier);

    pthread_exit(0);
}

void App::thread_image_acquisition(node_t node, pthread_barrier_t* init_barrier,
                                   cv::Ptr<cv::cuda::HOG_RT> gpu_hog, cv::HOGDescriptor cpu_hog,
                                   Mat* frames, struct task_info t_info, int graph_idx)
{
    fprintf(stdout, "node name: image_acquisition(source), task id: %d, node tid: %d\n", t_info.id, gettid());
#ifdef LOG_DEBUG
    char tabbuf[] = "\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t";
    tabbuf[node.node] = '\0';
#endif
    pgm_claim_node(node);

    edge_t *out_edge = (edge_t *)calloc(1, sizeof(edge_t));
    CheckError(pgm_get_edges_out(node, out_edge, 1));
    struct params_color_convert *out_buf = (struct params_color_convert *)pgm_get_edge_buf_p(*out_edge);
    if (out_buf == NULL)
        fprintf(stderr, "image acquisition out buffer is NULL\n");

    Size win_stride(args.win_stride_width, args.win_stride_height);
    Size win_size(args.win_width, args.win_width * 2);

    Mat* img = new Mat();
    vector<Rect>* found = new vector<Rect>();
    Mat* frame = new Mat();

    // Pre-allocate all gpu_img instances we will need (one per frame)
    unsigned cons_copies = 5; // be conservative so we don't overwrite anything
    cuda::GpuMat* gpu_img_array[cons_copies];
    for (unsigned i = 0; i < cons_copies; i++)
    {
        gpu_img_array[i] = new cuda::GpuMat();
    }

    *frame = frames[0];

    double level_scale[13];
    double scale_val = 1.0;
    for (unsigned i = 0; i < 13; i++)
    {
        level_scale[i] = scale_val;
        scale_val *= scale;
    }

    cv::cuda::Stream managed_stream;
    cuda::BufferPool pool(managed_stream);
    cuda::GpuMat* grad_array[cons_copies][13];
    cuda::GpuMat* qangle_array[cons_copies][13];
    cuda::GpuMat* block_hists_array[cons_copies][13];
    cuda::GpuMat* smaller_img_array[cons_copies][13];
    cuda::GpuMat* labels_array[cons_copies][13];
    for (unsigned i = 0; i < cons_copies; i++)
    {
        for (unsigned j = 0; j < 13; j++)
        {
            Size sz(cvRound(frame->cols / level_scale[j]), cvRound(frame->rows / level_scale[j]));

            grad_array[i][j] = new cuda::GpuMat();
            qangle_array[i][j] = new cuda::GpuMat();

            *grad_array[i][j]   = pool.getBuffer(sz, CV_32FC2);
            *qangle_array[i][j] = pool.getBuffer(sz, CV_8UC2);

            block_hists_array[i][j] = new cuda::GpuMat();

            *block_hists_array[i][j] = pool.getBuffer(1, gpu_hog->getTotalHistSize(sz), CV_32FC1);

            smaller_img_array[i][j] = new cuda::GpuMat();

            if (j != 0)
            {
                *smaller_img_array[i][j] = pool.getBuffer(sz, gpu_img_array[i]->type());
            }

            labels_array[i][j] = new cuda::GpuMat();

            *labels_array[i][j] = pool.getBuffer(1, gpu_hog->numPartsWithin(sz, win_size, win_stride).area(), CV_8UC1);
        }
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    gpu_hog->set_up_constants(stream);

    /* initialization is finished */
    pthread_barrier_wait(init_barrier);

    if (t_info.realtime) {
        // NOTE: LITMUS^RT initialization here is /almost/ identical to
        //       set_up_litmus_task(), but we don't allow for early releasing.
        // Handle signals locally. Deferring to our potentially non-real-time
        // parent may cause a priority inversion.
        struct sigaction handler;
        memset(&handler, 0, sizeof(handler));
        handler.sa_handler = gpu_hog->get_aborting_fz_sig_hndlr();
        sigaction(SIGSYS, &handler, NULL);
        // if (t_info.cluster != -1)
        //     CALL(be_migrate_to_domain(t_info.cluster));
        struct rt_task param;
        init_rt_task_param(&param);
        param.exec_cost = ms2ns(t_info.period) - 1;
        param.period = ms2ns(t_info.period);
        param.relative_deadline = ms2ns(t_info.relative_deadline);
        param.phase = ms2ns(t_info.phase);
        param.budget_policy = NO_ENFORCEMENT;
        param.cls = RT_CLASS_SOFT;
        param.priority = LITMUS_LOWEST_PRIORITY;
        if (t_info.cluster == -1)
        {
            param.cpu = 1; // default to 1, maybe ignored by GSN-EDF?
        }
        else if (t_info.sched == CONFIGURABLE) // the cluster is not -1
        {
            param.cpu = t_info.cluster;
        }
        else // the cluster is not -1 and it's not configurable, so migrate to that cluster
        {
            param.cpu = domain_to_first_cpu(t_info.cluster);
        }
        CALL( set_rt_task_param(gettid(), &param) );
        fprintf(stdout, "[%d | %d] Finished setting rt params.\n", gettid(), getpid());
        CALL( init_litmus() );
        fprintf(stdout, "[%d | %d] Called init_litmus.\n", gettid(), getpid());
        CALL( task_mode(LITMUS_RT_TASK) );
        fprintf(stdout, "[%d | %d] Now a real-time task.\n", gettid(), getpid());
        CALL( wait_for_ts_release() );
    }

    int count_frame = 0;
    while (count_frame < args.count / args.num_fine_graphs && running) {
        for (int j = graph_idx; j < 100; j += args.num_fine_graphs) {
            if (!t_info.realtime)
                usleep(30000);
            if (count_frame >= args.count / args.num_fine_graphs)
                break;

            /* choose the frame */
            *frame = frames[j];

            workBegin();
            lt_t frame_start_time = litmus_clock();

            /* Which of the (conservatively duplicated) data should we use? */
            unsigned data_idx = (j / args.num_fine_graphs) % cons_copies;
            cuda::GpuMat *gpu_img = gpu_img_array[data_idx];

            gpu_hog->is_aborting_frame = false;
            gpu_hog->setNumLevels(nlevels);
            gpu_hog->setHitThreshold(hit_threshold);
            gpu_hog->setScaleFactor(scale);
            gpu_hog->setGroupThreshold(gr_threshold);

            out_buf->frame = frame;
            out_buf->img = img;
            out_buf->gpu_img = gpu_img;
            out_buf->grad_array = grad_array[data_idx];
            out_buf->qangle_array = qangle_array[data_idx];
            out_buf->block_hists_array = block_hists_array[data_idx];
            out_buf->smaller_img_array = smaller_img_array[data_idx];
            out_buf->labels_array = labels_array[data_idx];
            out_buf->found = found;
            out_buf->img_to_show = img;
            out_buf->frame_index = j;
            out_buf->start_time = frame_start_time;
            CheckError(pgm_complete(node));

            found = new vector<Rect>();
            img = new Mat();
            count_frame++;
            if (t_info.realtime)
                sleep_next_period();
        }
    }

    delete frame;

    cudaStreamDestroy(stream);

    free(out_edge);
    CheckError(pgm_terminate(node));

    pthread_barrier_wait(init_barrier);

    CheckError(pgm_release_node(node));

    if (args.realtime)
        CALL( task_mode(BACKGROUND_TASK) );

    /* end is finished */
    pthread_barrier_wait(init_barrier);

    for (unsigned i = 0; i < cons_copies; i++)
    {
        gpu_img_array[i]->release();

        for (unsigned j = 0; j < 13; j++)
        {
            grad_array[i][j]->release();
            qangle_array[i][j]->release();
            block_hists_array[i][j]->release();
            smaller_img_array[i][j]->release();
            labels_array[i][j]->release();
        }
    }
}

void App::thread_color_convert(node_t node, pthread_barrier_t* init_barrier,
                               cv::Ptr<cv::cuda::HOG_RT> gpu_hog,
                               struct task_info t_info)
{
    fprintf(stdout, "node name: color_convert, task id: %d, node tid: %d\n", t_info.id, gettid());
#ifdef LOG_DEBUG
    char tabbuf[] = "\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t";
    tabbuf[node.node] = '\0';
#endif

    CheckError(pgm_claim_node(node));

    int ret = 0;

    edge_t *in_edge = (edge_t *)calloc(1, sizeof(edge_t));
    CheckError(pgm_get_edges_in(node, in_edge, 1));
    struct params_color_convert *in_buf = (struct params_color_convert *)pgm_get_edge_buf_c(*in_edge);
    if (in_buf == NULL)
        fprintf(stderr, "color convert node in buffer is NULL\n");

    edge_t *out_edge = (edge_t *)calloc(1, sizeof(edge_t));
    CheckError(pgm_get_edges_out(node, out_edge, 1));
    struct params_compute *out_buf = (struct params_compute *)pgm_get_edge_buf_p(*out_edge);
    if (out_buf == NULL)
        fprintf(stderr, "color convert node out buffer is NULL\n");

    Mat * frame;
    Mat * img;
    cuda::GpuMat * gpu_img;

    pthread_barrier_wait(init_barrier);

    struct rt_task param;
    int omlp_sem_od = -1;
    struct control_page* cp;
    if (t_info.realtime) {
        gpu_hog->set_up_litmus_task(t_info, param, &omlp_sem_od);
    }
    cp = get_ctrl_page();

    if(!hog_sample_errors)
    {
        cudaStream_t stream;
        cudaStreamCreate(&stream);

        do {
            ret = pgm_wait(node);
            if (gpu_hog->is_aborting_frame) {
                CheckError(pgm_complete(node));
                continue;
            }

            if(ret != PGM_TERMINATE)
            {
                CheckError(ret);
#ifdef LOG_DEBUG
                fprintf(stdout, "%s%d fires\n", tabbuf, node.node);
#endif

                frame = in_buf->frame;
                img = in_buf->img;
                gpu_img = in_buf->gpu_img;

                /* color convert node starts below */
                // Change format of the image
                if (true || make_gray) cvtColor(*frame, *img, COLOR_BGR2GRAY);
                else cvtColor(*frame, *img, COLOR_BGR2BGRA); // always using GPU

                // Perform HOG classification
                hogWorkBegin();

                /* =============
                * LOCK: upload image to GPU
                */
                gpu_hog->lock_fzlp(omlp_sem_od);
                gpu_hog->wait_forbidden_zone(omlp_sem_od, NODE_AB);
                lt_t fz_start = litmus_clock();

                gpu_img->upload(*img, stream);
                cp->fz_progress = FZ_POST_GPU_LAUNCH;
                exit_np();
                cudaStreamSynchronize(stream);
                gpu_hog->exit_forbidden_zone(omlp_sem_od);

                lt_t fz_len = litmus_clock() - fz_start;
                fprintf(stdout, "[%d | %d] Computation %d took %llu microseconds.\n",
                        gettid(), getpid(), NODE_AB, fz_len / 1000);
                gpu_hog->unlock_fzlp(omlp_sem_od);
                /*
                * UNLOCK: upload image to GPU
                * ============= */

                out_buf->gpu_img = gpu_img;
                out_buf->grad_array = in_buf->grad_array;
                out_buf->qangle_array = in_buf->qangle_array;
                out_buf->block_hists_array = in_buf->block_hists_array;
                out_buf->smaller_img_array = in_buf->smaller_img_array;
                out_buf->labels_array = in_buf->labels_array;
                out_buf->found = in_buf->found;
                out_buf->img_to_show = img;
                out_buf->frame_index = in_buf->frame_index;
                out_buf->start_time = in_buf->start_time;
                CheckError(pgm_complete(node));

                if (t_info.realtime)
                    sleep_next_period(); /* this calls the system call sys_complete_job. With early releasing, this shouldn't block.*/
            }
            else
            {
#ifdef LOG_DEBUG
                fprintf(stdout, "%s- %d terminates\n", tabbuf, node.node);
#endif
                //pgm_terminate(node);
            }

        } while(ret != PGM_TERMINATE);

        cudaStreamDestroy(stream);
    }

    pthread_barrier_wait(init_barrier);

    CheckError(pgm_release_node(node));

    free(in_edge);
    free(out_edge);

    if (t_info.realtime)
        CALL( task_mode(BACKGROUND_TASK) );

    /* end is finished */
    pthread_barrier_wait(init_barrier);

    pthread_exit(0);
}

static void sync_info_init(struct sync_info *s)
{
    s->start_time = 0;
    s->job_no = 0;
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

    cv::Ptr<cv::cuda::HOG_RT> gpu_hog = cv::cuda::HOG_RT::create(win_size, block_size, block_stride, cell_size, args.nbins);
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
        case CONFIGURABLE:
            sched_configurable_hog(gpu_hog, cpu_hog, frames);
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

void App::sched_configurable_hog(cv::Ptr<cv::cuda::HOG_RT> gpu_hog, cv::HOGDescriptor cpu_hog, Mat* frames)
{
    const std::vector< std::vector<node_config> > &level_configs = args.level_configurations;
    const std::vector<node_config> &source_config = args.source_configuration;
    const std::vector<node_config> &sink_config = args.sink_configuration;

    bool has_display_node = args.display || args.write_video;

    unsigned num_total_level_nodes = 0;
    for (unsigned i = 0; i < level_configs.size(); i++)
    {
        num_total_level_nodes += level_configs[i].size();
    }

    bool is_source_A = false;
    bool is_source_B = false;
    bool is_source_C = false;
    bool is_source_D = false;
    bool is_source_E = false;
    for (unsigned i = 0; i < source_config.size(); i++)
    {
        if (source_config[i] == NODE_A)
        {
            is_source_A = true;
        }
        else if (source_config[i] == NODE_AB)
        {
            is_source_B = true;
        }
        else if (source_config[i] == NODE_ABC)
        {
            is_source_C = true;
        }
        else if (source_config[i] == NODE_ABCD)
        {
            is_source_D = true;
        }
        else if (source_config[i] == NODE_ABCDE)
        {
            is_source_E = true;
        }
    }

    bool is_sink_A = false;
    bool is_sink_B = false;
    bool is_sink_C = false;
    bool is_sink_D = false;
    bool is_sink_E = false;
    for (unsigned i = 0; i < sink_config.size(); i++)
    {
        if (sink_config[i] == NODE_ABCDE)
        {
            is_sink_A = true;
        }
        else if (sink_config[i] == NODE_BCDE)
        {
            is_sink_B = true;
        }
        else if (sink_config[i] == NODE_CDE)
        {
            is_sink_C = true;
        }
        else if (sink_config[i] == NODE_DE)
        {
            is_sink_D = true;
        }
        else if (sink_config[i] == NODE_E)
        {
            is_sink_E = true;
        }
    }

    // Create the graphs
    char buf[30];
    sprintf(buf, "/tmp/graph_c%d_t%d", args.cluster, args.task_id);
    CheckError(pgm_init(buf, 1));

    /* graph construction */
    graph_t arr_arr_g [args.num_hog_inst][args.num_fine_graphs];

    pthread_barrier_t arr_arr_fine_init_barrier[args.num_hog_inst][args.num_fine_graphs];

    // Not all sync info structs will be used
    struct sync_info arr_arr_level_sync_info  [args.num_hog_inst][args.num_fine_graphs][NUM_SCALE_LEVELS][5];
    struct sync_info arr_arr_source_sync_info [args.num_hog_inst][args.num_fine_graphs];
    struct sync_info arr_arr_sink_sync_info   [args.num_hog_inst][args.num_fine_graphs];

    // Store pointers to the threads for joining later
    std::vector< std::vector< std::vector<thread *> > > all_threads;

    for (int inst_idx = 0; inst_idx < args.num_hog_inst; inst_idx++)
    {
        fprintf(stderr, "\nInitializing instance %d\n", inst_idx);

        /* graph construction */
        graph_t *arr_g = arr_arr_g[inst_idx];

        pthread_barrier_t *arr_fine_init_barrier = arr_arr_fine_init_barrier[inst_idx];

        // Sync info to keep job execution in order
        struct sync_info (*arr_level_sync_info)[NUM_SCALE_LEVELS][5] = arr_arr_level_sync_info[inst_idx];
        struct sync_info *arr_source_sync_info = arr_arr_source_sync_info[inst_idx];
        struct sync_info *arr_sink_sync_info = arr_arr_sink_sync_info[inst_idx];

        // Store pointers to the threads for joining later
        std::vector< std::vector<thread *> > inst_threads;

        for (int g_idx = 0; g_idx < args.num_fine_graphs; g_idx++)
        {
            fprintf(stderr, "Initializing graph %d\n", g_idx);

            pthread_barrier_t* fine_init_barrier = arr_fine_init_barrier + g_idx;

            // A graph consists of the graph itself, nodes, and edges
            graph_t* g_ptr = arr_g + g_idx;

            node_t image_acquisition_node;
            node_t color_convert_node;
            node_t compute_scales_node;
            node_t level_nodes [NUM_SCALE_LEVELS][5];
            node_t collect_locations_node;
            node_t display_node;

            edge_t ei_0;
            edge_t e0_1;
            edge_t level_edges [NUM_SCALE_LEVELS][6];
            edge_t e7_8;

            // Initialize the graph
            sprintf(buf, "hog_%d_%d", args.cluster, g_idx);
            CheckError(pgm_init_graph(g_ptr, buf));
            graph_t g = *g_ptr;

            // Initialize the nodes
            const char* level_node_names[] = { "node_1st", "node_2nd", "node_3rd", "node_4th", "node_5th" };
            if (!args.merge_color_convert)
            {
                CheckError(pgm_init_node(&image_acquisition_node, g, "image_acquisition"));
                CheckError(pgm_init_node(&color_convert_node, g, "color_convert"));
            }
            CheckError(pgm_init_node(&compute_scales_node, g, "compute_scales"));
            for (unsigned i = 0; i < NUM_SCALE_LEVELS; i++)
            {
                unsigned int num_nodes = level_configs[i].size();
                for (unsigned node_idx = 0; node_idx < num_nodes; node_idx++)
                {
                    CheckError(pgm_init_node(&(level_nodes[i][node_idx]), g, level_node_names[node_idx]));
                }
            }
            CheckError(pgm_init_node(&collect_locations_node, g, "collect_locations"));
            if (has_display_node)
            {
                CheckError(pgm_init_node(&display_node, g, "display"));
            }

            // Initialize the edges
            const char* level_edge_name_formats[] = { "e1_1st_%d", "e1st_2nd_%d", "e2nd_3rd_%d", "e3rd_4th_%d", "e4th_5th_%d", "e5th_7_%d" };
            edge_attr_t fast_mq_attr;
            memset(&fast_mq_attr, 0, sizeof(fast_mq_attr));
            fast_mq_attr.type = pgm_fast_fifo_edge;

            if (!args.merge_color_convert)
            {
                fast_mq_attr.nr_produce = sizeof(struct params_color_convert);
                fast_mq_attr.nr_consume = sizeof(struct params_color_convert);
                fast_mq_attr.nr_threshold = sizeof(struct params_color_convert);
                CheckError(pgm_init_edge(&ei_0, image_acquisition_node, color_convert_node, "ei_0", &fast_mq_attr));

                fast_mq_attr.nr_produce = sizeof(struct params_compute);
                fast_mq_attr.nr_consume = sizeof(struct params_compute);
                fast_mq_attr.nr_threshold = sizeof(struct params_compute);
                CheckError(pgm_init_edge(&e0_1, color_convert_node, compute_scales_node, "e0_1", &fast_mq_attr));
            }

            for (unsigned i = 0; i < NUM_SCALE_LEVELS; i++)
            {
                unsigned num_nodes = level_configs[i].size();
                size_t params_sizes[num_nodes + 1];

                for (unsigned edge_idx = 0; edge_idx < num_nodes; edge_idx++)
                {
                    switch (level_configs[i][edge_idx]) {
                        case NODE_A:
                        case NODE_AB:
                        case NODE_ABC:
                        case NODE_ABCD:
                        case NODE_ABCDE:
                            params_sizes[edge_idx] = sizeof(struct params_resize);
                            break;
                        case NODE_B:
                        case NODE_BC:
                        case NODE_BCD:
                        case NODE_BCDE:
                            params_sizes[edge_idx] = sizeof(struct params_compute_gradients);
                            break;
                        case NODE_C:
                        case NODE_CD:
                        case NODE_CDE:
                            params_sizes[edge_idx] = sizeof(struct params_compute_histograms);
                            break;
                        case NODE_D:
                        case NODE_DE:
                            params_sizes[edge_idx] = sizeof(struct params_fine_normalize);
                            break;
                        case NODE_E:
                            params_sizes[edge_idx] = sizeof(struct params_fine_classify);
                            break;
                        default:
                            break;
                    }
                }

                switch (sink_config[i])
                {
                    case NODE_ABCDE:
                        params_sizes[num_nodes] = sizeof(struct params_resize);
                        break;
                    case NODE_BCDE:
                        params_sizes[num_nodes] = sizeof(struct params_compute_gradients);
                        break;
                    case NODE_CDE:
                        params_sizes[num_nodes] = sizeof(struct params_compute_histograms);
                        break;
                    case NODE_DE:
                        params_sizes[num_nodes] = sizeof(struct params_fine_normalize);
                        break;
                    case NODE_E:
                        params_sizes[num_nodes] = sizeof(struct params_fine_classify);
                        break;
                    case NODE_NONE:
                        params_sizes[num_nodes] = sizeof(struct params_fine_collect_locations);
                        break;
                    default:
                        fprintf(stdout, "Invalid sink node configuration for level %d.\n", i);
                        break;
                }

                for (unsigned edge_idx = 0; edge_idx < num_nodes+1; edge_idx++)
                {
                    // Edge parameters: name, token counts
                    sprintf(buf, level_edge_name_formats[edge_idx], i);
                    fast_mq_attr.nr_produce = params_sizes[edge_idx];
                    fast_mq_attr.nr_consume = params_sizes[edge_idx];
                    fast_mq_attr.nr_threshold = params_sizes[edge_idx];

                    // Choose the nodes connected by the edge and initialize the edge
                    node_t node_start = edge_idx == 0         ? compute_scales_node    : level_nodes[i][edge_idx-1];
                    node_t node_end   = edge_idx == num_nodes ? collect_locations_node : level_nodes[i][edge_idx];
                    CheckError(pgm_init_edge(&(level_edges[i][edge_idx]),
                                                node_start, node_end,
                                                buf, &fast_mq_attr));
                }
            }

            if (has_display_node)
            {
                fast_mq_attr.nr_produce = sizeof(struct params_display);
                fast_mq_attr.nr_consume = sizeof(struct params_display);
                fast_mq_attr.nr_threshold = sizeof(struct params_display);
                CheckError(pgm_init_edge(&e7_8, collect_locations_node, display_node, "e7_8", &fast_mq_attr));
            }

            // Initialize the threads (one per node)
            if (args.merge_color_convert)
            {
                if (has_display_node)
                {
                    pthread_barrier_init(fine_init_barrier, 0, num_total_level_nodes + 3);
                }
                else
                {
                    pthread_barrier_init(fine_init_barrier, 0, num_total_level_nodes + 2);
                }
            }
            else
            {
                if (has_display_node)
                {
                    pthread_barrier_init(fine_init_barrier, 0, num_total_level_nodes + 5);
                }
                else
                {
                    pthread_barrier_init(fine_init_barrier, 0, num_total_level_nodes + 4);
                }
            }

            std::vector<thread *> graph_threads;

            struct sync_info (* in_sync_info)[5]  = arr_level_sync_info[((g_idx + args.num_fine_graphs - 1) % args.num_fine_graphs)];
            struct sync_info (* out_sync_info)[5] = arr_level_sync_info[g_idx];

            for (unsigned i = 0; i < level_configs.size(); i++)
            {
                for (unsigned node_idx = 0; node_idx < level_configs[i].size(); node_idx++)
                {
                    sync_info_init(&(in_sync_info[i][node_idx]));
                    sync_info_init(&(out_sync_info[i][node_idx]));
                }
            }

            struct sync_info in_source_sync_info  = arr_source_sync_info[((g_idx + args.num_fine_graphs - 1) % args.num_fine_graphs)];
            struct sync_info out_source_sync_info = arr_source_sync_info[g_idx];

            sync_info_init(&in_source_sync_info);
            sync_info_init(&out_source_sync_info);

            struct sync_info in_sink_sync_info  = arr_sink_sync_info[((g_idx + args.num_fine_graphs - 1) % args.num_fine_graphs)];
            struct sync_info out_sink_sync_info = arr_sink_sync_info[g_idx];

            sync_info_init(&in_sink_sync_info);
            sync_info_init(&out_sink_sync_info);

            // WCETs and response-time bounds for each node
            float bound_image_acquisition       = 33.0; // TODO
            float bound_color_convert           = args.non_level_bounds[0];
            float bound_compute_scales          = args.non_level_bounds[1];
            vector<vector<float>> bound_levels  = args.level_bounds;
            vector<vector<float>> cost_levels  = args.level_costs;

            /* | first graph release      | second graph release     | first graph release again
            *  <---------PERIOD--------->
            *  <--------------- PERIOD * args.num_fine_graphs ---------->
            */
            unsigned task_id = 0;
            int period = PERIOD * args.num_fine_graphs;
            struct task_info t_info;
            t_info.early = args.early;
            t_info.realtime = args.realtime;
            t_info.sched = CONFIGURABLE;
            t_info.period = period;
            t_info.relative_deadline = period; // use EDF
            t_info.source_config = &args.source_configuration;
            t_info.sink_config = &args.sink_configuration;
            t_info.has_display_node = has_display_node;
            if (args.cluster != -1)
                t_info.cluster = args.cluster;
            else
                t_info.cluster = args.cluster;

            t_info.phase = PERIOD * g_idx;
            t_info.id = task_id++;
            t_info.graph_idx = g_idx;
            if (!args.merge_color_convert)
            {
                thread *ti = new thread(&App::thread_image_acquisition, this,
                                        image_acquisition_node, fine_init_barrier,
                                        gpu_hog, cpu_hog, frames, t_info, g_idx);
                graph_threads.push_back(ti);

                // If the color-convert node is not merged, spawn its thread and the
                // compute-scale-levels thread separately
                t_info.id = task_id++;
                t_info.phase = t_info.phase + bound_image_acquisition;
                thread *t0 = new thread(&App::thread_color_convert, this,
                                        color_convert_node, fine_init_barrier, gpu_hog, t_info);
                graph_threads.push_back(t0);

                void* (cv::cuda::HOG_RT::* compute_scales_func)(node_t node, pthread_barrier_t* init_barrier, struct task_info t_info);
                if (is_source_E)
                {
                    compute_scales_func = &cv::cuda::HOG_RT::thread_fine_S_ABCDE;
                }
                else if (is_source_D)
                {
                    compute_scales_func = &cv::cuda::HOG_RT::thread_fine_S_ABCD;
                }
                else if (is_source_C)
                {
                    compute_scales_func = &cv::cuda::HOG_RT::thread_fine_S_ABC;
                }
                else if (is_source_B)
                {
                    compute_scales_func = &cv::cuda::HOG_RT::thread_fine_S_AB;
                }
                else if (is_source_A)
                {
                    compute_scales_func = &cv::cuda::HOG_RT::thread_fine_S_A;
                }
                else
                {
                    compute_scales_func = &cv::cuda::HOG_RT::thread_fine_compute_scales;
                }
                t_info.id = task_id++;
                t_info.phase = t_info.phase + bound_color_convert;
                t_info.s_info_in = &in_source_sync_info;
                t_info.s_info_out = &out_source_sync_info;
                thread *t1 = new thread(compute_scales_func, gpu_hog,
                                        compute_scales_node, fine_init_barrier, t_info);
                graph_threads.push_back(t1);
            }
            else
            {
                // Otherwise, just spawn the merged node
                t_info.s_info_in = &in_source_sync_info;
                t_info.s_info_out = &out_source_sync_info;
                thread *t1 = new thread(&App::thread_fine_CC_S_ABCDE, this,
                                        compute_scales_node, fine_init_barrier, gpu_hog,
                                        frames, t_info, g_idx);
                graph_threads.push_back(t1);
            }

            float level_start_phase = 0.0f;
            if (!args.merge_color_convert)
            {
                level_start_phase = PERIOD * g_idx + bound_image_acquisition + bound_color_convert + bound_compute_scales;
                printf("level start phase: %f\n", level_start_phase);
            }
            else
            {
                level_start_phase = PERIOD * g_idx + bound_compute_scales;
                printf("Level start phase: %f\n", level_start_phase);
            }
            float max_level_end_phase = level_start_phase;

            for (unsigned i = 0; i < NUM_SCALE_LEVELS; i++)
            {
                unsigned num_nodes = level_configs[i].size();

                if (num_nodes == 0)
                {
                    continue;
                }

                void* (cv::cuda::HOG_RT::* level_funcs[num_nodes])(node_t node, pthread_barrier_t* init_barrier, struct task_info t_info);

                for (unsigned node_idx = 0; node_idx < num_nodes; node_idx++)
                {
                    switch (level_configs[i][node_idx]) {
                        case NODE_A:
                            level_funcs[node_idx] = &cv::cuda::HOG_RT::thread_fine_resize;
                            break;
                        case NODE_B:
                            level_funcs[node_idx] = &cv::cuda::HOG_RT::thread_fine_compute_gradients;
                            break;
                        case NODE_C:
                            level_funcs[node_idx] = &cv::cuda::HOG_RT::thread_fine_compute_histograms;
                            break;
                        case NODE_D:
                            level_funcs[node_idx] = &cv::cuda::HOG_RT::thread_fine_normalize_histograms;
                            break;
                        case NODE_E:
                            level_funcs[node_idx] = &cv::cuda::HOG_RT::thread_fine_classify;
                            break;
                        case NODE_AB:
                            level_funcs[node_idx] = &cv::cuda::HOG_RT::thread_fine_AB;
                            break;
                        case NODE_BC:
                            level_funcs[node_idx] = &cv::cuda::HOG_RT::thread_fine_BC;
                            break;
                        case NODE_CD:
                            level_funcs[node_idx] = &cv::cuda::HOG_RT::thread_fine_CD;
                            break;
                        case NODE_DE:
                            level_funcs[node_idx] = &cv::cuda::HOG_RT::thread_fine_DE;
                            break;
                        case NODE_ABC:
                            level_funcs[node_idx] = &cv::cuda::HOG_RT::thread_fine_ABC;
                            break;
                        case NODE_BCD:
                            level_funcs[node_idx] = &cv::cuda::HOG_RT::thread_fine_BCD;
                            break;
                        case NODE_CDE:
                            level_funcs[node_idx] = &cv::cuda::HOG_RT::thread_fine_CDE;
                            break;
                        case NODE_ABCD:
                            level_funcs[node_idx] = &cv::cuda::HOG_RT::thread_fine_ABCD;
                            break;
                        case NODE_BCDE:
                            level_funcs[node_idx] = &cv::cuda::HOG_RT::thread_fine_BCDE;
                            break;
                        case NODE_ABCDE:
                            level_funcs[node_idx] = &cv::cuda::HOG_RT::thread_fine_ABCDE;
                            break;
                        default:
                            break;
                    }
                }

                for (unsigned node_idx = 0; node_idx < num_nodes; node_idx++)
                {
                    t_info.id = task_id++;
                    t_info.phase = node_idx == 0 \
                                        ? level_start_phase \
                                        : t_info.phase + bound_levels[i][node_idx - 1];
                    t_info.s_info_in = &(in_sync_info[i][node_idx]);
                    t_info.s_info_out = &(out_sync_info[i][node_idx]);
                    thread *tlevel = new thread(level_funcs[node_idx], gpu_hog,
                                                (level_nodes[i][node_idx]), fine_init_barrier, t_info);
                    graph_threads.push_back(tlevel);
                }

                float level_end_phase = t_info.phase + bound_levels[i][num_nodes-1];
                if (level_end_phase > max_level_end_phase)
                {
                    max_level_end_phase = level_end_phase;
                }
            }

            void* (cv::cuda::HOG_RT::* collect_locations_func)(node_t node, pthread_barrier_t* init_barrier, struct task_info t_info);
            if (is_sink_A)
            {
                collect_locations_func = &cv::cuda::HOG_RT::thread_fine_ABCDE_T;
            }
            else if (is_sink_B)
            {
                collect_locations_func = &cv::cuda::HOG_RT::thread_fine_BCDE_T;
            }
            else if (is_sink_C)
            {
                collect_locations_func = &cv::cuda::HOG_RT::thread_fine_CDE_T;
            }
            else if (is_sink_D)
            {
                collect_locations_func = &cv::cuda::HOG_RT::thread_fine_DE_T;
            }
            else if (is_sink_E)
            {
                collect_locations_func = &cv::cuda::HOG_RT::thread_fine_E_T;
            }
            else
            {
                collect_locations_func = &cv::cuda::HOG_RT::thread_fine_collect_locations;
            }
            t_info.id = task_id++;
            t_info.phase = max_level_end_phase;
            t_info.s_info_in = &in_sink_sync_info;
            t_info.s_info_out = &out_sink_sync_info;
            thread *t7 = new thread(collect_locations_func, gpu_hog,
                                    collect_locations_node, fine_init_barrier, t_info);
            graph_threads.push_back(t7);

            if (has_display_node)
            {
                thread *t8 = new thread(&App::thread_display, this,
                                        display_node, fine_init_barrier, g_idx == 0 && args.display);
                graph_threads.push_back(t8);
            }

            fprintf(stdout, "Created %d tasks\n", task_id);

            inst_threads.push_back(graph_threads);
        }

        all_threads.push_back(inst_threads);
    }

    /* graph construction finishes */

    printf("Joining pthreads...\n");

    for (int inst_idx = 0; inst_idx < args.num_hog_inst; inst_idx++)
    {
        printf("Joining for instance %d\n", inst_idx);

        graph_t *arr_g = arr_arr_g[inst_idx];

        for (int g_idx = 0; g_idx < args.num_fine_graphs; g_idx++)
        {
            printf("Joining for graph %d in instance %d\n", g_idx, inst_idx);

            graph_t g = arr_g[g_idx];

            std::vector<thread *> &graph_threads = all_threads[inst_idx][g_idx];
            for (unsigned t_idx = 0; t_idx < graph_threads.size(); t_idx++)
            {
                printf("Joining thread %d\n", t_idx);
                thread *t = graph_threads[t_idx];
                if (t->joinable()) t->join();
                // delete t;
            }
            for (unsigned t_idx = 0; t_idx < graph_threads.size(); t_idx++)
            {
                thread *t = graph_threads[t_idx];
                delete t;
            }

            CheckError(pgm_destroy_graph(g));
        }
    }

    //CheckError(pgm_destroy_graph(g));
    CheckError(pgm_destroy());
    fprintf(stdout, "cleaned up ...");
}

void App::thread_fine_CC_S_ABCDE(node_t node, pthread_barrier_t* init_barrier,
                                 cv::Ptr<cv::cuda::HOG_RT> gpu_hog, Mat* frames,
                                 struct task_info t_info, int graph_idx)
{
    fprintf(stdout, "node name: color_convert->classify_hists(source), task id: %d, node tid: %d\n", t_info.id, gettid());
#ifdef LOG_DEBUG
    char tabbuf[] = "\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t";
    tabbuf[node.node] = '\0';
#endif

    CheckError(pgm_claim_node(node));

    edge_t *out_edges = (edge_t *)calloc(NUM_SCALE_LEVELS, sizeof(edge_t));
    CheckError(pgm_get_edges_out(node, out_edges, NUM_SCALE_LEVELS));
    void **out_buf_ptrs = (void **)calloc(NUM_SCALE_LEVELS, sizeof(void *));
    for (int i = 0; i < NUM_SCALE_LEVELS; i++) {
        out_buf_ptrs[i] = (void *)pgm_get_edge_buf_p(out_edges[i]);
        if (out_buf_ptrs[i] == NULL)
            fprintf(stderr, "color_convert->classify_hists node out buffer is NULL\n");
    }

    // Color convert
    Size win_stride(args.win_stride_width, args.win_stride_height);
    Size win_size(args.win_width, args.win_width * 2);

    Mat img_aux;
    Mat* img = new Mat();
    Mat* img_to_show;
    vector<Rect>* found = new vector<Rect>();
    Mat frame;

    // Source (compute scale levels)

    // Pre-allocate all gpu_img instances we will need (one per frame)
    unsigned cons_copies = 5; // be conservative so we don't overwrite anything
    cuda::GpuMat* gpu_img_array[cons_copies];
    for (unsigned i = 0; i < cons_copies; i++)
    {
        gpu_img_array[i] = new cuda::GpuMat();
    }

    frame = frames[0];

    double level_scale[13];
    double scale_val = 1.0;
    for (unsigned i = 0; i < 13; i++)
    {
        level_scale[i] = scale_val;
        scale_val *= scale;
    }

    cv::cuda::Stream managed_stream;
    cuda::BufferPool pool(managed_stream);
    cuda::GpuMat* grad_array[cons_copies][13];
    cuda::GpuMat* qangle_array[cons_copies][13];
    cuda::GpuMat* block_hists_array[cons_copies][13];
    cuda::GpuMat* smaller_img_array[cons_copies][13];
    cuda::GpuMat* labels_array[cons_copies][13];
    for (unsigned i = 0; i < cons_copies; i++)
    {
        for (unsigned j = 0; j < 13; j++)
        {
            Size sz(cvRound(frame.cols / level_scale[j]), cvRound(frame.rows / level_scale[j]));

            grad_array[i][j] = new cuda::GpuMat();
            qangle_array[i][j] = new cuda::GpuMat();

            *grad_array[i][j]   = pool.getBuffer(sz, CV_32FC2);
            *qangle_array[i][j] = pool.getBuffer(sz, CV_8UC2);

            block_hists_array[i][j] = new cuda::GpuMat();

            *block_hists_array[i][j] = pool.getBuffer(1, gpu_hog->getTotalHistSize(sz), CV_32FC1);

            smaller_img_array[i][j] = new cuda::GpuMat();

            if (j != 0)
            {
                *smaller_img_array[i][j] = pool.getBuffer(sz, CV_8UC1); // HARD-CODED!!
            }

            labels_array[i][j] = new cuda::GpuMat();

            *labels_array[i][j] = pool.getBuffer(1, gpu_hog->numPartsWithin(sz, win_size, win_stride).area(), CV_8UC1);
        }
    }

    /* initialization is finished */
    pthread_barrier_wait(init_barrier);

    if (t_info.realtime) {
        // NOTE: LITMUS^RT initialization here is /almost/ identical to
        //       set_up_litmus_task(), but we don't allow for early releasing.
        // Handle signals locally. Deferring to our potentially non-real-time
        // parent may cause a priority inversion.
        struct sigaction handler;
        memset(&handler, 0, sizeof(handler));
        handler.sa_handler = gpu_hog->get_aborting_fz_sig_hndlr();
        sigaction(SIGSYS, &handler, NULL);
        // if (t_info.cluster != -1)
        //     CALL(be_migrate_to_domain(t_info.cluster));
        struct rt_task param;
        init_rt_task_param(&param);
        param.exec_cost = ms2ns(t_info.period) - 1;
        param.period = ms2ns(t_info.period);
        param.relative_deadline = ms2ns(t_info.relative_deadline);
        param.phase = ms2ns(t_info.phase);
        param.budget_policy = NO_ENFORCEMENT;
        param.cls = RT_CLASS_SOFT;
        param.priority = LITMUS_LOWEST_PRIORITY;
        if (t_info.cluster == -1)
        {
            param.cpu = 1; // default to 1, maybe ignored by GSN-EDF?
        }
        else if (t_info.sched == CONFIGURABLE) // the cluster is not -1
        {
            param.cpu = t_info.cluster;
        }
        else // the cluster is not -1 and it's not configurable, so migrate to that cluster
        {
            param.cpu = domain_to_first_cpu(t_info.cluster);
        }

        CALL( set_rt_task_param(gettid(), &param) );
        fprintf(stdout, "[%d | %d] Finished setting rt params.\n", gettid(), getpid());
        CALL( init_litmus() );
        fprintf(stdout, "[%d | %d] Called init_litmus.\n", gettid(), getpid());
        CALL( task_mode(LITMUS_RT_TASK) );
        fprintf(stdout, "[%d | %d] Now a real-time task.\n", gettid(), getpid());
        CALL( wait_for_ts_release() );
    }

    fprintf(stdout, "[%d | %d] Calling litmus_open_lock for OMLP_SEM.\n", gettid(), getpid());
    int omlp_sem_od = gpu_hog->open_lock(args.cluster); // use the cluster ID as the resource ID
    fprintf(stdout, "[%d | %d] Got OMLP_SEM=%d.\n", gettid(), getpid(), omlp_sem_od);
    struct control_page* cp = get_ctrl_page();

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    gpu_hog->set_up_constants(stream);

    int count_frame = 0;
    while (count_frame < args.count / args.num_fine_graphs && running)
    {
        for (int j = graph_idx; j < 100; j += args.num_fine_graphs)
        {
            if (!t_info.realtime)
                usleep(30000);
            if (count_frame >= args.count / args.num_fine_graphs)
                break;
            frame = frames[j];
            workBegin();
            lt_t frame_start_time = litmus_clock();

            /* ===========================
             * color convert
             */

            /* color convert node starts below */
            // Change format of the image
            if (make_gray) cvtColor(frame, img_aux, COLOR_BGR2GRAY);
            else if (use_gpu) cvtColor(frame, img_aux, COLOR_BGR2BGRA);
            else frame.copyTo(img_aux);

            // Resize image
            if (args.resize_src) resize(img_aux, *img, Size(args.width, args.height));
            else *img = img_aux;
            img_to_show = img;

            // Prep HOG classification
            hogWorkBegin();

            // Which of the (conservatively duplicated) data should we use?
            unsigned data_idx = (j / args.num_fine_graphs) % cons_copies;

            /* =============
             * LOCK: upload image to GPU
             */
            cuda::GpuMat *gpu_img = gpu_img_array[data_idx];

            SAMPLE_START_LOCK(lt_t fz_start, NODE_AB);

            gpu_img->upload(*img, stream);
            cp->fz_progress = FZ_POST_GPU_LAUNCH;
            exit_np();
            cudaStreamSynchronize(stream);
            gpu_hog->exit_forbidden_zone(omlp_sem_od);

            SAMPLE_STOP_LOCK(lt_t fz_len, NODE_AB);
            /*
             * UNLOCK: upload image to GPU
             * ============= */

            gpu_hog->is_aborting_frame = false;
            gpu_hog->setNumLevels(nlevels);
            gpu_hog->setHitThreshold(hit_threshold);
            gpu_hog->setScaleFactor(scale);
            gpu_hog->setGroupThreshold(gr_threshold);

            /*
             * end of color convert
             * =========================== */

            gpu_hog->fine_CC_S_ABCDE(t_info, out_buf_ptrs, gpu_img,
                                     grad_array[data_idx], qangle_array[data_idx],
                                     block_hists_array[data_idx],
                                     smaller_img_array[data_idx], labels_array[data_idx],
                                     found,
                                     img_to_show, j, stream, frame_start_time,
                                     omlp_sem_od);

            CheckError(pgm_complete(node));

            found = new vector<Rect>();
            img = new Mat();
            count_frame++;

            if (t_info.realtime)
                sleep_next_period();
        }
    }

    cudaStreamDestroy(stream);

    free(out_edges);
    free(out_buf_ptrs);

    CheckError(pgm_terminate(node));

    pthread_barrier_wait(init_barrier);

    CheckError(pgm_release_node(node));

    if (t_info.realtime)
        CALL( task_mode(BACKGROUND_TASK) );

    /* end is finished */
    pthread_barrier_wait(init_barrier);

    for (unsigned i = 0; i < cons_copies; i++)
    {
        gpu_img_array[i]->release();

        for (unsigned j = 0; j < 13; j++)
        {
            grad_array[i][j]->release();
            qangle_array[i][j]->release();
            block_hists_array[i][j]->release();
            smaller_img_array[i][j]->release();
            labels_array[i][j]->release();
        }
    }

    pthread_exit(0);
}
