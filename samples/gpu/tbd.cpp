#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <iomanip>
#include <stdexcept>
#include <map>
#include <opencv2/core/utility.hpp>
#include "opencv2/cudaobjdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/tbd.hpp"

using namespace std;
using namespace cv;
using namespace cv::tbd;

bool help_showed = false;

class Args;

void parseBboxFile(string bbox_filename, vector<vector<vector<double>>> &per_frame_bboxes, vector<vector<double>> &per_frame_camera_poses, vector<unsigned> &per_frame_history_choices, unsigned num_frames);
void parseDetections(vector<vector<vector<double>>> &perFrameBboxes, int frameId, vector<Detection> &detections, std::map<int, Trajectory> &trajectoryMap);

Point2d worldCoordsToScreenCoords(Vec3d &worldPos, Vec3d &cameraPos, Vec3d &cameraDir, int w, int h);

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

    // Configuration options for tracking
    double costOfNonAssignment;
    unsigned int timeWindowSize;
    unsigned int trackAgeThreshold;
    double trackVisibilityThreshold;
    double trackConfidenceThreshold;

    string history_distribution_string;
    vector<float> history_distribution;

    bool track_pedestrians;
    string pedestrian_bbox_filename;
    bool track_vehicles;
    string vehicle_bbox_filename;

    bool write_tracking;
    string pedestrian_tracking_filepath;
    string vehicle_tracking_filepath;

    int num_tracking_iters;
    int num_tracking_frames;

private:
    void parseHistoryDistribution(char *dist_arg);
};

class App
{
public:
    App(const Args& s);
    void run();

    void handleKey(char key);

    void performTrackingStep(Tracker *tracker,
                             vector<Detection> &foundDetections,
                             std::map<int, Trajectory> &trajectoryMap,
                             bool shouldStoreMetrics);

    void writeTrackingOutputToFile(Tracker *tracker,
                                   vector<unsigned> &historyAges,
                                   std::map<int, Trajectory> &trajectoryMap,
                                   string filepath);

    void hogWorkBegin();
    void hogWorkEnd();
    string hogWorkFps() const;

    void workBegin();
    void workEnd();
    string workFps() const;

    string frameIndex(int) const;

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
    double hog_work_fps;

    int64 work_begin;
    double work_fps;

    unsigned frame_id;

    bool use_provided_history;
    vector<unsigned> provided_history_choices;
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
         << "  [--history_distribution <string>] # comma-separated distribution of age of history for tracking (e.g., '0.7,0.3' for 70% prior frame, 30% two prior)\n"
         << "  [--pedestrian_bbox_filename <string>] # filename of pedestrian bounding box results for ground truth"
         << "  [--vehicle_bbox_filename <string>] # filename of vehicle bounding box results for ground truth"
         << "  [--write_tracking <bool>] # writer tracking output or not"
         << "  [--tracking_filename <string>] # tracking output filename\n"
         << "  [--num_tracking_iters <int>] # number of times to repeat the tracking experiment"
         << "  [--num_tracking_frames <int>] # number of frames of the video to track\n";
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

    // Configuration options for tracking
    costOfNonAssignment = 10.0;
    timeWindowSize = 16;//16
    trackAgeThreshold = 4;//8;
    trackVisibilityThreshold = 0.3; //0.4;//0.6;
    trackConfidenceThreshold = 0.2;

    track_pedestrians = false;
    track_vehicles = false;

    num_tracking_iters = 1;
    num_tracking_frames = 100;

    // Age of prior results
    history_distribution = std::vector<float>(1, 1.0); // default to using the previous frame
}

void Args::parseHistoryDistribution(char *dist_arg)
{
    this->history_distribution_string = dist_arg;

    vector<float> dist;

    // Parse the comma-separate distribution information
    string dist_str = string(dist_arg);
    size_t prev_pos = 0;
    size_t pos = 0;
    do
    {
        pos = dist_str.find(",", prev_pos);

        float val = stof(dist_str.substr(prev_pos, pos));
        dist.push_back(val);

        prev_pos = pos + 1;
    }
    while (pos != string::npos);

    // Normalize it (sum should be 1.0)
    float total = 0.0f;
    for (unsigned i = 0; i < dist.size(); i++)
    {
        total += dist[i];
    }

    for (unsigned i = 0; i < dist.size(); i++)
    {
        dist[i] /= total;
    }

    this->history_distribution = dist;
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
        else if (string(argv[i]) == "--history_distribution") { args.parseHistoryDistribution(argv[++i]);
        }
        else if (string(argv[i]) == "--pedestrian_bbox_filename") { args.pedestrian_bbox_filename = argv[++i]; args.track_pedestrians = true; }
        else if (string(argv[i]) == "--vehicle_bbox_filename") { args.vehicle_bbox_filename = argv[++i]; args.track_vehicles = true; }
        else if (string(argv[i]) == "--write_tracking") args.write_tracking = (string(argv[++i]) == "true");
        else if (string(argv[i]) == "--pedestrian_tracking_filepath") args.pedestrian_tracking_filepath = argv[++i];
        else if (string(argv[i]) == "--vehicle_tracking_filepath") args.vehicle_tracking_filepath = argv[++i];
        else if (string(argv[i]) == "--num_tracking_iters") args.num_tracking_iters = atoi(argv[++i]);
        else if (string(argv[i]) == "--num_tracking_frames") args.num_tracking_frames = atoi(argv[++i]);
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

    use_provided_history = false;
    provided_history_choices = vector<unsigned>();

    // cout << "Scale: " << scale << endl;
    // if (args.resize_src)
    //     cout << "Resized source: (" << args.width << ", " << args.height << ")\n";
    // cout << "Group threshold: " << gr_threshold << endl;
    // cout << "Levels number: " << nlevels << endl;
    // cout << "Win size: (" << args.win_width << ", " << args.win_width*2 << ")\n";
    // cout << "Win stride: (" << args.win_stride_width << ", " << args.win_stride_height << ")\n";
    // cout << "Block size: (" << args.block_width << ", " << args.block_width << ")\n";
    // cout << "Block stride: (" << args.block_stride_width << ", " << args.block_stride_height << ")\n";
    // cout << "Cell size: (" << args.cell_width << ", " << args.cell_width << ")\n";
    // cout << "Bins number: " << args.nbins << endl;
    // cout << "Hit threshold: " << hit_threshold << endl;
    // cout << "Gamma correction: " << gamma_corr << endl;
    // cout << endl;
}


void App::run()
{
    running = true;
    cv::VideoWriter video_writer;

    Size win_stride(args.win_stride_width, args.win_stride_height);
    Size win_size(args.win_width, args.win_width * 2);
    Size block_size(args.block_width, args.block_width);
    Size block_stride(args.block_stride_width, args.block_stride_height);
    Size cell_size(args.cell_width, args.cell_width);

    cv::Ptr<cv::cuda::HOG> gpu_hog = cv::cuda::HOG::create(win_size, block_size, block_stride, cell_size, args.nbins);
    cv::HOGDescriptor cpu_hog(win_size, block_size, block_stride, cell_size, args.nbins);

    if(args.svm_load) {
        // std::vector<float> svm_model;
        // const std::string model_file_name = args.svm;
        // FileStorage ifs(model_file_name, FileStorage::READ);
        // if (ifs.isOpened()) {
        //     ifs["svm_detector"] >> svm_model;
        // } else {
        //     const std::string what =
        //             "could not load model for hog classifier from file: "
        //             + model_file_name;
        //     throw std::runtime_error(what);
        // }

        // // check if the variables are initialized
        // if (svm_model.empty()) {
        //     const std::string what =
        //             "HoG classifier: svm model could not be loaded from file"
        //             + model_file_name;
        //     throw std::runtime_error(what);
        // }

        // gpu_hog->setSVMDetector(svm_model);
        // cpu_hog.setSVMDetector(svm_model);

        const std::string model_file_name = args.svm;
        const std::string obj_name = "";
        bool loaded_cpu = cpu_hog.load(model_file_name);
        bool loaded_gpu = gpu_hog->load(model_file_name);
        if (!loaded_cpu || !loaded_gpu)
        {
            const std::string what =
                    "could not load model for hog classifier from file: "
                    + model_file_name;
            throw std::runtime_error(what);
        }
    } else {
        // Create HOG descriptors and detectors here
        Mat detector = gpu_hog->getDefaultPeopleDetector();

        gpu_hog->setSVMDetector(detector);
        cpu_hog.setSVMDetector(detector);
    }

    gr_threshold = gpu_hog->getGroupThreshold();
    cpu_hog.nlevels = 15;
    gpu_hog->setNumLevels(15);
    nlevels = gpu_hog->getNumLevels();
    scale = gpu_hog->getScaleFactor();
    gpu_hog->setHitThreshold(0.45);
    hit_threshold = gpu_hog->getHitThreshold();
    gamma_corr = gpu_hog->getGammaCorrection();

    cout << "Scale: " << scale << endl;
    if (args.resize_src)
        cout << "Resized source: (" << args.width << ", " << args.height << ")\n";
    cout << "Group threshold: " << gr_threshold << endl;
    cout << "Levels number: " << nlevels << endl;
    cout << "Win size: (" << cpu_hog.winSize.width << ", " << cpu_hog.winSize.height << ")\n";
    cout << "Win stride: (" << args.win_stride_width << ", " << args.win_stride_height << ")\n";
    cout << "Block size: (" << cpu_hog.blockSize.width << ", " << cpu_hog.blockSize.height << ")\n";
    cout << "Block stride: (" << cpu_hog.blockStride.width << ", " << cpu_hog.blockStride.height << ")\n";
    cout << "Cell size: (" << cpu_hog.cellSize.width << ", " << cpu_hog.cellSize.height << ")\n";
    cout << "Bins number: " << cpu_hog.nbins << endl;
    cout << "Hit threshold: " << hit_threshold << endl;
    cout << "Gamma correction: " << gamma_corr << endl;
    cout << endl << "History age distribution: ";
    for (unsigned i = 0; i < args.history_distribution.size(); i++)
    {
        if (i > 0) { cout << ", "; }
        cout << i+1 << "=" << args.history_distribution[i];
    }
    cout << endl;
    cout << endl;

    cout << "gpusvmDescriptorSize : " << gpu_hog->getDescriptorSize()
         << endl;
    cout << "cpusvmDescriptorSize : " << cpu_hog.getDescriptorSize()
         << endl;

    // If a file of ground-truth bounding boxes has been supplied, pre-process it
    vector<vector<vector<double>>> pedestrian_per_frame_bboxes;
    vector<vector<double>> per_frame_camera_poses; // x, y, yaw or x, y, z, pitch, yaw, roll
    if (!args.pedestrian_bbox_filename.empty())
    {
        parseBboxFile(args.pedestrian_bbox_filename, pedestrian_per_frame_bboxes, per_frame_camera_poses, this->provided_history_choices, args.num_tracking_frames);

        if (!this->provided_history_choices.empty())
        {
            this->use_provided_history = true;
        }
    }

    vector<vector<vector<double>>> vehicle_per_frame_bboxes;
    if (!args.vehicle_bbox_filename.empty())
    {
        parseBboxFile(args.vehicle_bbox_filename, vehicle_per_frame_bboxes, per_frame_camera_poses, this->provided_history_choices, args.num_tracking_frames);

        if (!this->provided_history_choices.empty())
        {
            this->use_provided_history = true;
        }
    }

    int iteration = 0;

    while (running && iteration < args.num_tracking_iters)
    {
        VideoCapture vc;
        Mat frame;
        vector<String> filenames;

        std::map<int, Trajectory> pedestrianTrajectoryMap;
        std::map<int, Trajectory> vehicleTrajectoryMap;

        std::vector<std::vector<Track>> pedestrianTrackOutputBuffer;
        std::vector<std::vector<Track>> vehicleTrackOutputBuffer;
        for (unsigned i = 0; i < args.history_distribution.size(); i++)
        {
            pedestrianTrackOutputBuffer.push_back(std::vector<Track>());
            vehicleTrackOutputBuffer.push_back(std::vector<Track>());
        }

        // Store results for outputting later
        std::vector<unsigned> historyAges;

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

        Mat img_aux, img, img_to_show;
        cuda::GpuMat gpu_img;

        TbdArgs tbdArgs(this->args.costOfNonAssignment,
                             this->args.timeWindowSize,
                             this->args.trackAgeThreshold,
                             this->args.trackVisibilityThreshold,
                             this->args.trackConfidenceThreshold);
        Tracker pedestrianTracker(&tbdArgs);
        Tracker vehicleTracker(&tbdArgs);

        this->frame_id = 0;

        // Iterate over all frames
        while (running && !frame.empty() && this->frame_id < args.num_tracking_frames)
        {
            workBegin();

            // Change format of the image
            if (make_gray) cvtColor(frame, img_aux, COLOR_BGR2GRAY);
            else if (use_gpu) cvtColor(frame, img_aux, COLOR_BGR2BGRA);
            else frame.copyTo(img_aux);

            // Resize image
            if (args.resize_src) resize(img_aux, img, Size(args.width, args.height));
            else img = img_aux;
            img_to_show = img;

            vector<Detection> pedestrianDetections;
            vector<Detection> vehicleDetections;

            // If a ground-truth file was provided, use that instead of HOG
            if (!pedestrian_per_frame_bboxes.empty() || !vehicle_per_frame_bboxes.empty())
            {
                parseDetections(pedestrian_per_frame_bboxes, this->frame_id, pedestrianDetections, pedestrianTrajectoryMap);
                parseDetections(vehicle_per_frame_bboxes, this->frame_id, vehicleDetections, vehicleTrajectoryMap);
            }
            else
            {
                vector<Rect> foundRects;

                // Perform HOG classification
                hogWorkBegin();
                if (use_gpu)
                {
                    gpu_img.upload(img);
                    gpu_hog->setNumLevels(nlevels);
                    gpu_hog->setHitThreshold(hit_threshold);
                    gpu_hog->setWinStride(win_stride);
                    gpu_hog->setScaleFactor(scale);
                    gpu_hog->setGroupThreshold(gr_threshold);
                    gpu_hog->detectMultiScale(gpu_img, foundRects);
                }
                else
                {
                    cpu_hog.nlevels = nlevels;
                    cpu_hog.detectMultiScale(img, foundRects, hit_threshold, win_stride,
                                            Size(0, 0), scale, gr_threshold);
                }
                hogWorkEnd();

                for (unsigned rect_idx = 0; rect_idx < foundRects.size(); rect_idx++)
                {
                    // Create a detection object with no known ID and unknown confidence
                    // TODO: actually pass confidence vector to detectMultiScale
                    Detection d(-1, this->frame_id, foundRects[rect_idx], -1.0);
                    pedestrianDetections.push_back(d);
                }
            }

            /* ==========================
            *   Beginning of tracking
            * =========================== */

            // Reset the tracks at the beginning of the input
            if (this->frame_id == 0)
            {
                for (unsigned buf_idx = 0; buf_idx < args.history_distribution.size(); buf_idx++)
                {
                    pedestrianTrackOutputBuffer[buf_idx].clear();
                    vehicleTrackOutputBuffer[buf_idx].clear();
                }

                pedestrianTracker.reset();
                vehicleTracker.reset();

                historyAges.empty();
            }

            // Choose which prior results to use
            unsigned historyAge = 0; // invalid, must be at least 1
            if (this->use_provided_history)
            {
                historyAge = this->provided_history_choices[this->frame_id];
            }
            else
            {
                float cumulativeDist = 0.0f;
                float r = ((float)rand()) / RAND_MAX;
                for (unsigned i = 0; i < args.history_distribution.size(); i++)
                {
                    cumulativeDist += args.history_distribution[i];
                    if (r < cumulativeDist)
                    {
                        historyAge = i+1;
                        break;
                    }
                }
                if (historyAge == 0) { historyAge = args.history_distribution.size(); }
            }
            historyAges.push_back(historyAge);

            // Retrieve the prior track information
            std::vector<Track> pedestrianTracks;
            std::vector<Track> vehicleTracks;
            if (this->frame_id >= historyAge)
            {
                unsigned priorTrackIndex = (this->frame_id - historyAge) % args.history_distribution.size();
                std::vector<Track> priorPedestrianTracks = pedestrianTrackOutputBuffer[priorTrackIndex];
                std::vector<Track> priorVehicleTracks = vehicleTrackOutputBuffer[priorTrackIndex];

                for (unsigned tidx = 0; tidx < priorPedestrianTracks.size(); tidx++)
                {
                    pedestrianTracks.push_back(Track(priorPedestrianTracks[tidx]));
                }
                for (unsigned tidx = 0; tidx < priorVehicleTracks.size(); tidx++)
                {
                    vehicleTracks.push_back(Track(priorVehicleTracks[tidx]));
                }
            }

            pedestrianTracker.setTracks(pedestrianTracks);
            vehicleTracker.setTracks(vehicleTracks);

            if (args.track_pedestrians)
            {
                performTrackingStep(&pedestrianTracker, pedestrianDetections,
                                    pedestrianTrajectoryMap, true /* store metrics */);
            }
            if (args.track_vehicles)
            {
                performTrackingStep(&vehicleTracker, vehicleDetections,
                                    vehicleTrajectoryMap, true /* store metrics */);
            }

            // Store the tracking results
            pedestrianTracks = pedestrianTracker.getTracks();
            vehicleTracks = vehicleTracker.getTracks();
            unsigned currentTrackIndex = this->frame_id % args.history_distribution.size();
            pedestrianTrackOutputBuffer[currentTrackIndex] = pedestrianTracks;
            vehicleTrackOutputBuffer[currentTrackIndex] = vehicleTracks;

            /*
            * end of tracking
            * =========================== */

            // Draw positive classified windows
            for (size_t i = 0; i < pedestrianDetections.size(); i++)
            {
                Rect r = pedestrianDetections[i].bbox;
                rectangle(img_to_show, r.tl(), r.br(), Scalar(255, 0, 0), 3);
            }

            for (size_t i = 0; i < vehicleDetections.size(); i++)
            {
                Rect r = vehicleDetections[i].bbox;
                rectangle(img_to_show, r.tl(), r.br(), Scalar(0, 255, 0), 3);
            }

            // If tracking was successful, draw rectangles for tracking
            if (pedestrianTracks.size() + vehicleTracks.size() > 0)
            {
                for (unsigned int trackIdx = 0; trackIdx < pedestrianTracks.size() + vehicleTracks.size(); trackIdx++)
                {
                    Track *track;
                    if (trackIdx < pedestrianTracks.size())
                    {
                        track = &((pedestrianTracks)[trackIdx]);
                    }
                    else
                    {
                        track = &((vehicleTracks)[trackIdx - pedestrianTracks.size()]);
                    }

                    // Don't draw tracks that are too new and/or with too low confidence
                    // TODO: remove check for maxConfidence >= when HOG gives confidence
                    if ((track->age < args.trackAgeThreshold && track->maxConfidence >= 0.0 && track->maxConfidence < args.trackConfidenceThreshold) ||
                        (track->age < args.trackAgeThreshold / 2))
                    {
                        continue;
                    }

                    // double opacity = ((track->avgConfidence / 3.0 > 1.0) ? track->avgConfidence / 3.0 : 1.0) < 0.5 ?
                    //                 ((track->avgConfidence / 3.0 > 1.0) ? track->avgConfidence / 3.0 : 1.0) :
                    //                 0.5;

                    // Draw the centroids of prior positions; if the camera poses are known,
                    // convert track positions from world coords to screen coords, otherwise
                    // just use the bounding box centers directly in screen space
                    if (per_frame_camera_poses.size() > 0)
                    {
                        for (unsigned int pidx = 0; pidx < track->worldCoords.size() - 1; pidx++)
                        {
                            if (track->scores[pidx] == 0.0)
                            {
                                continue;
                            }

                            int centroid_alpha = (int)((float)(track->frames[pidx]) / this->frame_id * 255);
                            Scalar color = Scalar(track->color[0], track->color[1], track->color[2], centroid_alpha);

                            Vec3d centroid = track->worldCoords[pidx];

                            vector<double> &camera_pose = per_frame_camera_poses[this->frame_id];
                            Vec3d cameraPos, cameraDir;
                            if (camera_pose.size() == 3)
                            {
                                // x, y, and yaw are provided (assume z, pitch, and roll are 0)
                                cameraPos = Vec3d(camera_pose[0], camera_pose[1], 0.0);
                                cameraDir = Vec3d(0.0, camera_pose[2], 0.0);
                            }
                            else
                            {
                                // x, y, z, and pitch, yaw, roll
                                cameraPos = Vec3d(camera_pose[0], camera_pose[1], camera_pose[2]);
                                cameraDir = Vec3d(camera_pose[3], camera_pose[4], camera_pose[5]);
                            }

                            Point2d screenCoord = worldCoordsToScreenCoords(centroid, cameraPos, cameraDir, frame.cols, frame.rows);

                            if (screenCoord.x >= 0 && screenCoord.x <= frame.cols &&
                                screenCoord.y >= 0 && screenCoord.y <= frame.rows)
                            {
                                circle(img_to_show, screenCoord, 4, color, CV_FILLED);
                            }
                        }
                    }
                    else
                    {
                        for (unsigned int bboxIdx = 0; bboxIdx < track->bboxes.size() - 1; bboxIdx++)
                        {
                            int centroid_alpha = (int)((float)(track->frames[bboxIdx]) / this->frame_id * 255);
                            Scalar color = Scalar(track->color[0], track->color[1], track->color[2], centroid_alpha);

                            Rect *bbox = &(track->bboxes[bboxIdx]);
                            Point2d centroid(bbox->x + bbox->width/2, bbox->y + bbox->height/2);
                            circle(img_to_show, centroid, 2, color, CV_FILLED);
                        }
                    }

                    // Draw the rectangle
                    Rect r = track->bboxes.back();
                    rectangle(img_to_show, r.tl(), r.br(), track->color, 2);

                    // Display the average confidence
                    Point textPoint(track->bboxes.back().x + track->bboxes.back().width / 2,
                                    track->bboxes.back().y + track->bboxes.back().height / 2);
                    //putText(img_to_show, confidenceText(track->avgConfidence), textPoint, FONT_HERSHEY_SIMPLEX, 1., track->color, 2);
                }
            }

            if (use_gpu)
                putText(img_to_show, "Mode: GPU", Point(5, 25), FONT_HERSHEY_SIMPLEX, 1., Scalar(255, 100, 0), 2);
            else
                putText(img_to_show, "Mode: CPU", Point(5, 25), FONT_HERSHEY_SIMPLEX, 1., Scalar(255, 100, 0), 2);
            putText(img_to_show, "FPS HOG: " + hogWorkFps(), Point(5, 65), FONT_HERSHEY_SIMPLEX, 1., Scalar(255, 100, 0), 2);
            putText(img_to_show, "FPS total: " + workFps(), Point(5, 105), FONT_HERSHEY_SIMPLEX, 1., Scalar(255, 100, 0), 2);
            putText(img_to_show, "Frame: " + frameIndex(this->frame_id), Point(5, 145), FONT_HERSHEY_SIMPLEX, 1., Scalar(255, 100, 0), 2);
            if (!args.history_distribution_string.empty())
            {
                putText(img_to_show, "History distribution: " + args.history_distribution_string, Point(5, 185), FONT_HERSHEY_SIMPLEX, 1., Scalar(255, 100, 0), 2);
            }
            imshow("opencv_gpu_hog", img_to_show);

            // cout << "About to read in the next video frame" << endl;

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
                                      img_to_show.size(), true);
                    if (!video_writer.isOpened())
                        throw std::runtime_error("can't create video writer");
                }

                if (make_gray) cvtColor(img_to_show, img, COLOR_GRAY2BGR);
                else cvtColor(img_to_show, img, COLOR_BGRA2BGR);

                video_writer << img;
            }

            handleKey((char)waitKey(3));

            this->frame_id++;
        }


        if (args.write_tracking)
        {
            if (!args.pedestrian_tracking_filepath.empty())
            {
                writeTrackingOutputToFile(&pedestrianTracker, historyAges, pedestrianTrajectoryMap, args.pedestrian_tracking_filepath);
            }

            if (!args.vehicle_tracking_filepath.empty())
            {
                writeTrackingOutputToFile(&vehicleTracker, historyAges, vehicleTrajectoryMap, args.vehicle_tracking_filepath);
            }
        }

        iteration++;
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
        hit_threshold+=0.05;
        cout << "Hit threshold: " << hit_threshold << endl;
        break;
    case 'r':
    case 'R':
        hit_threshold = max(0.0, hit_threshold - 0.05);
        cout << "Hit threshold: " << hit_threshold << endl;
        break;
    case 'c':
    case 'C':
        gamma_corr = !gamma_corr;
        cout << "Gamma correction: " << gamma_corr << endl;
        break;
    }
}

void App::performTrackingStep(Tracker *tracker,
                              vector<Detection> &foundDetections,
                              std::map<int, Trajectory> &trajectoryMap,
                              bool shouldStoreMetrics)
{
    // Predict the new locations of the tracks
    tracker->predictNewLocationsOfTracks(this->frame_id);

    // Filter out tracks with predictions that are out of the window
    tracker->filterTracksOutOfBounds(0, 1280, 0, 720);

    // Use predicted track positions to map current detections to existing tracks
    std::vector<int> assignments;
    std::vector<unsigned int> unassignedTracks, unassignedDetections;
    tracker->detectionToTrackAssignment(foundDetections, assignments,
                                        unassignedTracks, unassignedDetections,
                                        false, this->frame_id);

    // Update the tracks (assigned and unassigned)
    tracker->updateAssignedTracks(foundDetections, assignments);
    tracker->updateUnassignedTracks(unassignedTracks, this->frame_id);

    // Update trajectories for assigned detections/tracks
    unsigned numAssigned = 0;
    vector<Track> tracks = tracker->getTracks();
    for (unsigned trackIdx = 0; trackIdx < tracks.size(); trackIdx++)
    {
        if (assignments[trackIdx] < 0) { continue; }

        numAssigned++;

        unsigned detectionIdx = assignments[trackIdx];

        Track *track = &(tracks[trackIdx]);
        Detection *detection = &(foundDetections[detectionIdx]);

        // Ignore detections for which no ground truth information exists
        if (detection->id < 0) { continue; }

        Trajectory *trajectory = &(trajectoryMap[detection->id]);
        trajectory->addTrackingInfo(this->frame_id, track);
    }

    // Update trajectories for unassigned detections - note that this
    // will miss the first position/frame associated with each track,
    // because they are created just after
    for (unsigned udix = 0; udix < unassignedDetections.size(); udix++)
    {
        unsigned detectionIdx = unassignedDetections[udix];
        Detection *detection = &(foundDetections[detectionIdx]);

        // Ignore detections for which no ground truth information exists
        if (detection->id < 0) { continue; }

        Trajectory *trajectory = &(trajectoryMap[detection->id]);
        trajectory->addTrackingInfo(this->frame_id, NULL);
    }

    // Delete any tracks that are lost enough, and create new tracks
    // for unassigned detections
    tracker->deleteLostTracks();
    tracker->createNewTracks(foundDetections, unassignedDetections);

    if (shouldStoreMetrics)
    {
        tracker->truePositives.push_back(numAssigned);
        tracker->falseNegatives.push_back(unassignedDetections.size());
        tracker->falsePositives.push_back(unassignedTracks.size());
        tracker->groundTruths.push_back(foundDetections.size());
        tracker->numMatches.push_back(numAssigned); // not necessarily the same as TP_t

        double bboxOverlap = 0.0;
        for (unsigned tidx = 0; tidx < tracks.size(); tidx++)
        {
            bboxOverlap += tracks[tidx].bboxOverlap;
        }
        tracker->bboxOverlap.push_back(bboxOverlap);
    }
}

void App::writeTrackingOutputToFile(Tracker *tracker,
                                    vector<unsigned> &historyAges,
                                    std::map<int, Trajectory> &trajectoryMap,
                                    string filepath)
{
    ofstream tracking_file;
    tracking_file.open(filepath, ios::out | ios::app);

    // Write selected history ages
    tracking_file << "history|";
    for (unsigned i = 0; i < historyAges.size(); i++)
    {
        if (i > 0)
        {
            tracking_file << ",";
        }

        tracking_file << historyAges[i];
    }

    tracking_file << std::endl;

    // Derive trajectory-based metrics
    std::vector<int> idSwapsPerFrame = std::vector<int>(this->frame_id, 0);
    std::map<int, int> numFragmentationsPerTrack;
    int numMostlyTracked = 0;
    int numPartiallyTracked = 0;
    int numMostlyLost = 0;
    std::map<int, Trajectory>::iterator traj_it;
    for (traj_it = trajectoryMap.begin(); traj_it != trajectoryMap.end(); ++traj_it)
    {
        Trajectory *trajectory = &(traj_it->second);

        numFragmentationsPerTrack[trajectory->id] = 0;

        // An ID switch (added to IDSW) occurs when a ground-truth object is matched to
        // some track j and the last known assignment was track k != j.
        // A fragmentation is counted each time a trajectory changes its status
        // from tracked to untracked and tracking of that same trajectory is resumed
        // at a later frame.
        bool isNew = true;
        bool prevTracked = false;
        int prevTrackId = -1;
        int numTrackedFrames = 0;
        for (unsigned pfid = 0; pfid < trajectory->presentFrames.size(); pfid++)
        {
            int fnum = trajectory->presentFrames[pfid];

            // Check for ID swaps
            if (trajectory->isTrackedPerFrame[fnum])
            {
                int trackId = trajectory->trackIdPerFrame[fnum];
                if (isNew)
                {
                    prevTrackId = trackId;
                }
                else if (trackId != prevTrackId)
                {
                    std::cout << "[frame " << fnum << "] target " << trajectory->id << " switched from track " << prevTrackId << " to track " << trackId << std::endl;
                    idSwapsPerFrame[fnum]++;
                    prevTrackId = trackId;
                }
            }

            // Count the number of frames tracked
            if (trajectory->isTrackedPerFrame[fnum])
            {
                numTrackedFrames++;
            }

            // Check for fragmentations
            if (!isNew && !prevTracked && trajectory->isTrackedPerFrame[fnum])
            {
                numFragmentationsPerTrack[trajectory->id]++;
            }

            prevTracked = trajectory->isTrackedPerFrame[fnum];
            isNew = pfid == 0;
        }

        // A trajectory is mostly tracked if it is tracked at least 80%
        // of its lifetime, and mostly lost if it is tracked at most 20%
        // of its lifetime
        double trackedRatio = ((double)numTrackedFrames) / trajectory->presentFrames.size();
        if (trackedRatio >= 0.8)
        {
            numMostlyTracked++;
        }
        else if (trackedRatio > 0.2)
        {
            numPartiallyTracked++;
        }
        else
        {
            numMostlyLost++;
        }
    }

    // Write the tracking information to the output file
    for (traj_it = trajectoryMap.begin(); traj_it != trajectoryMap.end(); ++traj_it)
    {
        Trajectory *trajectory = &(traj_it->second);

        // Write the trajectory's object ID,
        tracking_file << "object|" << trajectory->id << "|";

        // tracking info each frame,
        for (int pfid = 0; pfid < trajectory->presentFrames.size(); pfid++)
        {
            if (pfid > 0)
            {
                tracking_file << ";";
            }

            int fnum = trajectory->presentFrames[pfid];

            tracking_file << fnum << "," << trajectory->isTrackedPerFrame[fnum];
            tracking_file << "," << trajectory->trackIdPerFrame[fnum];
        }

        tracking_file << "|";

        // and fragmentations
        tracking_file << "FM," << numFragmentationsPerTrack[trajectory->id];

        tracking_file << std::endl;
    }

    // Write per-frame tracking evaluation metrics to the output file
    double totalBboxOverlap = 0.0;
    for (unsigned fnum = 0; fnum < tracker->truePositives.size(); fnum++)
    {
        // Write the frame number
        tracking_file << "frame|" << fnum << "|";

        // Write the metrics for the frame
        tracking_file << "TP," << tracker->truePositives[fnum] << ";";
        tracking_file << "FN," << tracker->falseNegatives[fnum] << ";";
        tracking_file << "FP," << tracker->falsePositives[fnum] << ";";
        tracking_file << "GT," << tracker->groundTruths[fnum] << ";";
        tracking_file << "c," << tracker->numMatches[fnum] << ";";
        tracking_file << "IDSW," << idSwapsPerFrame[fnum] << ";";
        tracking_file << "sum_di," << tracker->bboxOverlap[fnum] << std::endl;

        totalBboxOverlap += tracker->bboxOverlap[fnum];
    }

    // Compute MOTA, A-MOTA, and MOTP for the scenario
    double motaNumerator = 0.0;
    double amotaNumerator = 0.0;
    double motaDenominator = 0.0;
    double motpNumerator = totalBboxOverlap;
    double motpDenominator = 0.0;
    for (unsigned fnum = 0; fnum < tracker->truePositives.size(); fnum++)
    {
        motaNumerator += (tracker->falseNegatives[fnum] + tracker->falsePositives[fnum] + idSwapsPerFrame[fnum]);
        amotaNumerator += (tracker->falseNegatives[fnum] + tracker->falsePositives[fnum]);
        motaDenominator += tracker->groundTruths[fnum];
        motpDenominator += tracker->numMatches[fnum];
    }

    double mota = 1 - (motaNumerator / motaDenominator);
    double amota = 1 - (amotaNumerator / motaDenominator);
    double motp = motpNumerator / motpDenominator;

    // Write total scenario tracking evaluation metrics (MT, ML, MOTA, MOTP)
    tracking_file << "scenario|MT," << numMostlyTracked << ";";
    tracking_file << "PT," << numPartiallyTracked << ";";
    tracking_file << "ML," << numMostlyLost << ";";
    tracking_file << "MOTA," << mota << ";";
    tracking_file << "A-MOTA," << amota << ";";
    tracking_file << "MOTP," << motp << std::endl;

    tracking_file.close();
}


inline void App::hogWorkBegin() { hog_work_begin = getTickCount(); }

inline void App::hogWorkEnd()
{
    int64 delta = getTickCount() - hog_work_begin;
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

inline string App::frameIndex(int i) const
{
    stringstream ss;
    ss << i;
    return ss.str();
}

void parseBboxFile(string bbox_filename, vector<vector<vector<double>>> &per_frame_bboxes, vector<vector<double>> &per_frame_camera_poses, vector<unsigned> &per_frame_history_choices, unsigned num_frames)
{
    bool shouldParseCameraPoses = per_frame_camera_poses.empty();

    ifstream bbox_file(bbox_filename);

    int prev_frame = -1;
    int start_frame = -1;
    vector<vector<double>> current_frame_bboxes;

    string line = "";
    while (getline(bbox_file, line))
    {
        if (line.find("|") == string::npos)
        {
            continue;
        }

        size_t prev_pos = 0;
        size_t pos = 0;

        // Check if the history choices have already been made (good for replaying)
        if (per_frame_history_choices.empty() && line.find("history") != string::npos)
        {
            pos = line.find("|");
            prev_pos = pos + 1;

            do
            {
                pos = line.find(",", prev_pos);

                int val = stoul(line.substr(prev_pos, pos));
                per_frame_history_choices.push_back(val);

                prev_pos = pos + 1;
            }
            while (pos != string::npos);

            continue;
        }

        // Get the frame number
        pos = line.find("|");
        prev_pos = pos + 1;
        int frame_num = stoi(line.substr(0, pos));

        // Check if it is ground-truth data or a real detection
        size_t num_seps = std::count(line.begin(), line.end(), '|');
        bool is_ground_truth = num_seps > 4;

        // Check if this is the first line read (assume they're in order by frame)
        if (start_frame == -1)
        {
            // If it's a real detection, start at 0, otherwise use
            // the frame number provided
            start_frame = is_ground_truth ? frame_num : 0;
            prev_frame = start_frame;
        }

        // If this is the first line we've seen from this frame, start a new vector
        // (there might have been multiple frames without any detections)
        for (int i = prev_frame; i < frame_num; i++)
        {
            per_frame_bboxes.push_back(current_frame_bboxes);
            current_frame_bboxes = vector<vector<double>>();
        }

        // The number of separators determines whether this file contains
        // ground truth or the output of external detections;
        // if external detections, there are no object IDs
        int objId;
        if (!is_ground_truth)
        {
            // External detections
            objId = -2; // unknown
        }
        else
        {
            // Ground truth, go ahead and get the object ID
            pos = line.find("|", prev_pos);
            objId = stoi(line.substr(prev_pos, pos));
            prev_pos = pos + 1;
        }

        // If the ID is -1, this is the camera's pose
        if (shouldParseCameraPoses && objId == -1)
        {
            vector<double> camera_pose;

            do
            {
                pos = line.find("|", prev_pos);

                double val = stod(line.substr(prev_pos, pos));
                camera_pose.push_back(val);

                prev_pos = pos + 1;
            }
            while (pos != string::npos);

            per_frame_camera_poses.push_back(camera_pose);
        }
        else if (objId != -1)
        {
            // Otherwise, get the info from this bounding box into a vector
            vector<double> bbox_info;
            bbox_info.push_back(objId);

            do
            {
                pos = line.find("|", prev_pos);

                double val = stod(line.substr(prev_pos, pos));
                bbox_info.push_back(val);

                prev_pos = pos + 1;
            }
            while (pos != string::npos);

            // Add this bounding box to this frame's list
            current_frame_bboxes.push_back(bbox_info);
        }

        prev_frame = frame_num;
    }

    // Add the last frame's bounding boxes (and any that were missed)
    for (int i = prev_frame; i < start_frame + num_frames; i++)
    {
        per_frame_bboxes.push_back(current_frame_bboxes);
        current_frame_bboxes = vector<vector<double>>();
    }
}

void parseDetections(vector<vector<vector<double>>> &perFrameBboxes, int frameId, vector<Detection> &detections, std::map<int, Trajectory> &trajectoryMap)
{
    if (perFrameBboxes.empty())
    {
        return;
    }

    vector<vector<double>> bounding_boxes = perFrameBboxes[frameId];

    for (unsigned bb_idx = 0; bb_idx < bounding_boxes.size(); bb_idx++)
    {
        vector<double> bbox = bounding_boxes[bb_idx];

        int objectId = bbox[0];
        Rect r(bbox[1], bbox[3], bbox[2] - bbox[1], bbox[4] - bbox[3]);

        Detection d(objectId, frameId, r, 1.0);

        Vec3d worldPos;
        if (bbox.size() > 5)
        {
            double xPos = bbox[5];
            double yPos = bbox[6];
            double zPos = (bbox.size() >= 8) ? bbox[7] : 0.0; // assume 0 if not provided
            worldPos = Vec3d(xPos, yPos, zPos);
        }
        d.worldPosition = worldPos;

        detections.push_back(d);

        // If the detection is ground-truth information, update the position
        // of the trajectory, creating it if necessary
        if (objectId >= 0)
        {
            if (trajectoryMap.find(objectId) == trajectoryMap.end())
            {
                Trajectory t(objectId);
                trajectoryMap[objectId] = t;
            }

            trajectoryMap[objectId].addPosition(frameId, r, d.worldPosition);
        }
    }
}

Point2d worldCoordsToScreenCoords(Vec3d &worldPos, Vec3d &cameraPos, Vec3d &cameraDir, int w, int h)
{
    double cameraPitch = cameraDir[0];
    double cameraYaw = cameraDir[1];
    double cameraRoll = cameraDir[2];

    // Transform worldPos from the world frame to the camera frame
    double cp = cos(cameraPitch * CV_PI / 180.0);
    double sp = sin(cameraPitch * CV_PI / 180.0);
    double cy = cos(cameraYaw * CV_PI / 180.0);
    double sy = sin(cameraYaw * CV_PI / 180.0);
    double cr = cos(cameraRoll * CV_PI / 180.0);
    double sr = sin(cameraRoll * CV_PI / 180.0);
    double w2c_data[] = { cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr, cameraPos[0],
                          sy * cp, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr, cameraPos[1],
                               sp,               -cp * sr,                 cp * cr, cameraPos[2],
                                0,                      0,                       0,            1 };
    Mat cameraToWorld = Mat(4, 4, CV_64F, w2c_data).clone();
    Mat4d worldToCamera = cameraToWorld.inv();

    double xCameraFrame = worldToCamera.at<double>(0,0) * worldPos[0] + worldToCamera.at<double>(0,1) * worldPos[1] + worldToCamera.at<double>(0,2) * worldPos[2] + worldToCamera.at<double>(0,3) * 1.0;
    double yCameraFrame = worldToCamera.at<double>(1,0) * worldPos[0] + worldToCamera.at<double>(1,1) * worldPos[1] + worldToCamera.at<double>(1,2) * worldPos[2] + worldToCamera.at<double>(1,3) * 1.0;
    double zCameraFrame = worldToCamera.at<double>(2,0) * worldPos[0] + worldToCamera.at<double>(2,1) * worldPos[1] + worldToCamera.at<double>(2,2) * worldPos[2] + worldToCamera.at<double>(2,3) * 1.0;

    // The camera frame has x forward, y to the right, and z up;
    // we want x to the right, y down, and z forward
    double x = yCameraFrame;
    double y = -zCameraFrame;
    double z = xCameraFrame;

    // Filter out positions behind the camera
    if (z < 0)
    {
        return Point2d(-1, -1);
    }

    // Multiply the camera calibration matrix by these coordinates to handle angle, etc.
    double calib_val = w / (2.0 * tan(90.0 * CV_PI / 360.0));
    double camera_calibration_data[] = { calib_val,         0, w / 2.0,
                                                 0, calib_val, h / 2.0,
                                                 0,         0,     1.0 };
    Mat cameraCalib = Mat(3, 3, CV_64F, camera_calibration_data).clone();

    double xCalib = cameraCalib.at<double>(0,0) * x + cameraCalib.at<double>(0,1) * y + cameraCalib.at<double>(0,2) * z;
    double yCalib = cameraCalib.at<double>(1,0) * x + cameraCalib.at<double>(1,1) * y + cameraCalib.at<double>(1,2) * z;
    double zCalib = cameraCalib.at<double>(2,0) * x + cameraCalib.at<double>(2,1) * y + cameraCalib.at<double>(2,2) * z;

    // Finally, divide by the z-coordinate to handle the distance to the camera plane
    int xPixel = round(xCalib / zCalib);
    int yPixel = round(yCalib / zCalib);

    return Point2d(xPixel, yPixel);
}