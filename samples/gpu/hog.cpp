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

using namespace std;
using namespace cv;

bool help_showed = false;

class Track;

double computeBoundingBoxOverlap(Rect &predBbox, Rect &bbox);

/*
 * A trajectory corresponds to the ground-truth sequence of positions
 * of a tracked object.
 */
class Trajectory
{
public:
    Trajectory(); // necessary to put it in a map
    Trajectory(int id);

    int id;

    vector<int> presentFrames;
    map<int, Rect> positionPerFrame;

    // Tracking information per frame, assumed to exist for a given key (frame ID)
    // only if isTrackedPerFrame[key] == true
    map<int, bool> isTrackedPerFrame;
    map<int, int> trackIdPerFrame;
    map<int, Rect> predPosPerFrame;
    map<int, Rect> trackPosPerFrame;
    map<int, double> bboxOverlapPerFrame;

    int getFirstFrame();

    void addPosition(int frame, Rect &bbox);
    void addTrackingInfo(int frame, Track *track);
};

/*
 * A detection corresponds to an observed (potentially incorrectly)
 * position of an object of interest.
 */
class Detection
{
public:
    Detection(int id, Rect &bbox, double confidence);

    int id; // -1 if not known

    Rect bbox;

    double confidence; // between 0.0 and 1.0
};

/*
 * A track correpsonds to a sequence of detections matched
 * together over time.
 */
class Track
{
public:
    Track(Rect &bbox, double score);

    unsigned int id;

    Scalar color;

    std::vector<Rect> bboxes;
    std::vector<double> scores;

    Point2d (*motionModel)(Track &track);

    unsigned int age;
    unsigned int totalVisibleCount;

    double maxConfidence;
    double avgConfidence;

    Rect predPosition;
};

// Assign IDs to tracks
unsigned int nextTrackId = 0;

Point2d constantVelocityMotionModel(Track &track);

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

    string bbox_filename;

    bool write_tracking;
    string tracking_filename;
};


class App
{
public:
    App(const Args& s);
    void run();

    void handleKey(char key);

    void hogWorkBegin();
    void hogWorkEnd();
    string hogWorkFps() const;

    void workBegin();
    void workEnd();
    string workFps() const;

    string frameIndex(int) const;

    string message() const;

    // Tracking functions
    void predictNewLocationsOfTracks(vector<Track> &tracks);

    void detectionToTrackAssignment(vector<Detection> &detections, vector<Track> &tracks, vector<int> &assignments, vector<unsigned int> &unassignedTracks, vector<unsigned int> &unassignedDetections, cv::Ptr<cv::cuda::HOG> gpu_hog, bool debug);

    void updateAssignedTracks(vector<Track> &tracks, vector<Detection> &detections, vector<double> &confidenceScores, vector<int> &assignments);
    void updateUnassignedTracks(vector<Track> &tracks, vector<unsigned int> &unassignedTracks);
    void deleteLostTracks(vector<Track> &tracks);
    void createNewTracks(vector<Track> &tracks, vector<Detection> &detections, vector<unsigned int> &unassignedDetections, vector<double> &confidenceScores);

    // Helper functions
    inline bool equalsZero(double val)
    {
        // return val == 0.0;
        return (val < 0.0) ? (val > -0.00000001) : (val < 0.00000001);
    }

    void calculateCostMatrix(vector<Track> &tracks, vector<Detection> &detections, vector<vector<double>> &costMatrix);
    void classifyAssignments(vector<unsigned int> &assignmentPerRow, unsigned int numTracks, unsigned int numDetections,
                            vector<int> &assignments, vector<unsigned int> &unassignedTracks, vector<unsigned int> &unassignedDetections);
    void solveAssignmentProblem(vector<vector<double>> &costMatrix, unsigned int numTracks, unsigned int numDetections,
                                vector<int> &assignments, vector<unsigned int> &unassignedTracks, vector<unsigned int> &unassignedDetections,
                                bool debug);
    void updateTrackConfidence(Track *track);

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
         << "  [--bbox_filename <string>] # filename of bounding box results for ground truth"
         << "  [--write_tracking <bool>] # writer tracking output or not"
         << "  [--tracking_filename <string>] # tracking output filename\n";
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
    timeWindowSize = 16;
    trackAgeThreshold = 4;//8;
    trackVisibilityThreshold = 0.3; //0.4;//0.6;
    trackConfidenceThreshold = -1.0;
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
        else if (string(argv[i]) == "--bbox_filename") args.bbox_filename = argv[++i];
        else if (string(argv[i]) == "--write_tracking") args.write_tracking = (string(argv[++i]) == "true");
        else if (string(argv[i]) == "--tracking_filename") args.tracking_filename = argv[++i];
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
    nlevels = gpu_hog->getNumLevels();
    scale = gpu_hog->getScaleFactor();
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
    cout << "Cell size: (" << args.cell_width << ", " << args.cell_width << ")\n";
    cout << "Bins number: " << args.nbins << endl;
    cout << "Hit threshold: " << hit_threshold << endl;
    cout << "Gamma correction: " << gamma_corr << endl;
    cout << endl;

    cout << "gpusvmDescriptorSize : " << gpu_hog->getDescriptorSize()
         << endl;
    cout << "cpusvmDescriptorSize : " << cpu_hog.getDescriptorSize()
         << endl;

    // If a file of ground-truth bounding boxes has been supplied, pre-process it
    vector<vector<vector<int>>> per_frame_bboxes;
    if (!args.bbox_filename.empty())
    {
        ifstream bbox_file(args.bbox_filename);

        int prev_frame = 0;
        int start_frame = 0;
        vector<vector<int>> current_frame_bboxes;

        string line = "";
        while (getline(bbox_file, line))
        {
            if (line.find("|") == string::npos)
            {
                continue;
            }

            // Get the info from this bounding box into a vector
            vector<int> bbox_info;

            size_t prev_pos = 0;
            size_t pos = 0;

            // Get the frame number
            pos = line.find("|");
            prev_pos = pos + 1;
            int frame_num = stoi(line.substr(0, pos));

            do
            {
                pos = line.find("|", prev_pos);

                int val = stoi(line.substr(prev_pos, pos));
                bbox_info.push_back(val);

                prev_pos = pos + 1;
            }
            while (pos != string::npos);

            // Check if this is the first line read (assume they're in order by frame)
            if (start_frame == 0)
            {
                start_frame = frame_num;
                prev_frame = frame_num;
            }

            // If this is the first bounding box we've seen from this frame, start a new vector
            // (there might have been multiple frames without any detections)
            for (int i = prev_frame; i < frame_num; i++)
            {
                per_frame_bboxes.push_back(current_frame_bboxes);
                current_frame_bboxes = vector<vector<int>>();
            }

            // Add this bounding box to this frame's list
            current_frame_bboxes.push_back(bbox_info);

            prev_frame = frame_num;
        }

        cout << "Found bounding boxes starting at frame " << start_frame;
        cout << " and going through frame " << prev_frame << endl;

        cout << "First BBOX for vehicle ID=" << per_frame_bboxes[0][0][0] << endl;
        cout << "Last frame contained " << per_frame_bboxes[per_frame_bboxes.size() - 1].size() << " vehicles" << endl;
    }

    std::map<int, Trajectory> trajectoryMap;
    std::vector<Track> tracks;

    bool first_pass = true;
    std::vector<int> truePositives;  // TP_t: # assigned detections
    std::vector<int> falseNegatives; // FN_t: # unmatched detections
    std::vector<int> falsePositives; // FP_t: # unmatched tracks (hypotheses)
    std::vector<int> groundTruths;   // GT_t: # objects in the scene
    std::vector<int> numMatches;     // c_t:  # matches

    while (running)
    {
        VideoCapture vc;
        Mat frame;
        vector<String> filenames;

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

        this->frame_id = 0;

        // Iterate over all frames
        while (running && !frame.empty())
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

            vector<Detection> foundDetections;

            // If a ground-truth file was provided, use that instead of HOG
            if (!per_frame_bboxes.empty())
            {
                vector<vector<int>> bounding_boxes = per_frame_bboxes[this->frame_id];

                for (unsigned bb_idx = 0; bb_idx < bounding_boxes.size(); bb_idx++)
                {
                    vector<int> bbox = bounding_boxes[bb_idx];

                    int objectId = bbox[0];
                    Rect r(bbox[1], bbox[3], bbox[2] - bbox[1], bbox[4] - bbox[3]);

                    Detection d(objectId, r, 1.0);
                    foundDetections.push_back(d);

                    // Update the position of the trajectory, creating it if necessary
                    if (trajectoryMap.find(objectId) == trajectoryMap.end())
                    {
                        Trajectory t(objectId);
                        trajectoryMap[objectId] = t;
                    }

                    trajectoryMap[objectId].addPosition(this->frame_id, r);
                }
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
                    Detection d(-1, foundRects[rect_idx], 0.0);
                }
            }

            /* ==========================
            *   Beginning of tracking
            * =========================== */

            // Reset the tracks at the beginning of the input
            if (this->frame_id == 0)
            {
                tracks.clear();
            }

            // Predict the new locations of the tracks
            App::predictNewLocationsOfTracks(tracks);

            std::vector<double> confidences;

            // Use predicted track positions to map current detections to existing tracks
            std::vector<int> assignments;
            std::vector<unsigned int> unassignedTracks, unassignedDetections;
            App::detectionToTrackAssignment(foundDetections, tracks, assignments,
                                            unassignedTracks, unassignedDetections, gpu_hog,
                                            false);

            // Update the tracks (assigned and unassigned)
            App::updateAssignedTracks(tracks, foundDetections, confidences, assignments);
            App::updateUnassignedTracks(tracks, unassignedTracks);

            // Update trajectories for assigned detections/tracks
            unsigned numAssigned = 0;
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
            App::deleteLostTracks(tracks);
            App::createNewTracks(tracks, foundDetections, unassignedDetections, confidences);

            if (first_pass)
            {
                truePositives.push_back(numAssigned);
                falseNegatives.push_back(unassignedDetections.size());
                falsePositives.push_back(unassignedTracks.size());
                groundTruths.push_back(foundDetections.size());
                numMatches.push_back(numAssigned); // not necessarily the same as TP_t
            }

            /*
            * end of tracking
            * =========================== */

            // Draw positive classified windows
            // for (size_t i = 0; i < found.size(); i++)
            // {
            //     Rect r = found[i];
            //     rectangle(img_to_show, r.tl(), r.br(), Scalar(0, 255, 0), 3);
            // }

            // If tracking was successful, draw rectangles for tracking
            if (tracks.size() > 0)
            {
                for (unsigned int trackIdx = 0; trackIdx < tracks.size(); trackIdx++)
                {
                    Track *track = &((tracks)[trackIdx]);

                    // Don't draw tracks that are too new and/or with too low confidence
                    if ((track->age < args.trackAgeThreshold && track->maxConfidence < args.trackConfidenceThreshold) ||
                        (track->age < args.trackAgeThreshold / 2))
                    {
                        continue;
                    }

                    // double opacity = ((track->avgConfidence / 3.0 > 1.0) ? track->avgConfidence / 3.0 : 1.0) < 0.5 ?
                    //                 ((track->avgConfidence / 3.0 > 1.0) ? track->avgConfidence / 3.0 : 1.0) :
                    //                 0.5;

                    // Draw the centroid of prior rectangles
                    for (unsigned int bboxIdx = 0; bboxIdx < track->bboxes.size() - 1; bboxIdx++)
                    {
                        int centroid_alpha = (int)((float)bboxIdx / track->bboxes.size() * 255);
                        Scalar color = Scalar(track->color[0], track->color[1], track->color[2], centroid_alpha);

                        Rect *bbox = &(track->bboxes[bboxIdx]);
                        Point2d centroid(bbox->x + bbox->width/2, bbox->y + bbox->height/2);
                        rectangle(img_to_show, centroid, centroid, color, 2);
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
            imshow("opencv_gpu_hog", img_to_show);

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

        if (first_pass && args.write_tracking && !args.tracking_filename.empty())
        {
            ofstream tracking_file;
            tracking_file.open(args.tracking_filename, ios::out | ios::app);

            // Derive trajectory-based metrics
            std::vector<int> idSwapsPerFrame = std::vector<int>(this->frame_id, 0);
            int numMostlyTracked = 0;
            int numMostlyLost = 0;
            std::map<int, int> numFragmentationsPerTrack;
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
                            idSwapsPerFrame[fnum]++;
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
                else if (trackedRatio <= 0.2)
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

                    if (trajectory->isTrackedPerFrame[fnum])
                    {
                        Rect &predBbox = trajectory->predPosPerFrame[fnum];
                        tracking_file << ",PREDBBOX[" << predBbox.x << "^" << predBbox.y << "^";
                        tracking_file << predBbox.width << "^" << predBbox.height << "]";

                        Rect &trBbox = trajectory->trackPosPerFrame[fnum];
                        tracking_file << ",TRBBOX[" << trBbox.x << "^" << trBbox.y << "^";
                        tracking_file << trBbox.width << "^" << trBbox.height << "]";

                        tracking_file << ",d[" << trajectory->bboxOverlapPerFrame[fnum] << "]";
                    }
                }

                tracking_file << "|";

                // and fragmentations
                tracking_file << "FM," << numFragmentationsPerTrack[trajectory->id];

                tracking_file << std::endl;
            }

            // Write per-frame tracking evaluation metrics to the output file
            for (unsigned fnum = 0; fnum < truePositives.size(); fnum++)
            {
                // Write the frame number
                tracking_file << "frame|" << fnum << "|";

                // Write the metrics for the frame
                tracking_file << "TP," << truePositives[fnum] << ";";
                tracking_file << "FN," << falseNegatives[fnum] << ";";
                tracking_file << "FP," << falsePositives[fnum] << ";";
                tracking_file << "GT," << groundTruths[fnum] << ";";
                tracking_file << "c," << numMatches[fnum] << ";";
                tracking_file << "IDSW," << idSwapsPerFrame[fnum] << std::endl;
            }

            // Write total scenario tracking evaluation metrics
            tracking_file << "scenario|MT," << numMostlyTracked << ";";
            tracking_file << "scenario|ML," << numMostlyLost << std::endl;

            tracking_file.close();
        }

        first_pass = false;
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

Trajectory::Trajectory()
{
    this->id = -1;
}

Trajectory::Trajectory(int id)
{
    this->id = id;
}

int Trajectory::getFirstFrame()
{
    if (this->presentFrames.size() == 0)
    {
        return -1;
    }
    else
    {
        return this->presentFrames[0];
    }
}

void Trajectory::addPosition(int frame, Rect &bbox)
{
    this->presentFrames.push_back(frame);
    this->positionPerFrame[frame] = bbox;
}

void Trajectory::addTrackingInfo(int frame, Track *track)
{
    if (track)
    {
        this->isTrackedPerFrame[frame] = true;
        this->trackIdPerFrame[frame] = track->id;

        Rect &bbox = track->bboxes[track->bboxes.size() - 1];
        Rect &predBbox = track->predPosition;
        Rect &gtPos = this->positionPerFrame[frame];
        this->trackPosPerFrame[frame] = bbox;
        this->predPosPerFrame[frame] = predBbox;
        this->bboxOverlapPerFrame[frame] = computeBoundingBoxOverlap(bbox, gtPos);
    }
    else
    {
        this->isTrackedPerFrame[frame] = false;
    }
}

Detection::Detection(int id, Rect &bbox, double confidence)
{
    this->id = id;
    this->bbox = bbox;
    this->confidence = confidence;
}

void App::predictNewLocationsOfTracks(vector<Track> &tracks)
{
    for (unsigned int i = 0; i < tracks.size(); i++)
    {
        // Get the last bounding box on this track
        Rect &bbox = tracks[i].bboxes.back();

        // Predict tthe current location of the track
        Point2d predictedCentroid = tracks[i].motionModel(tracks[i]);

        // Shift the bounding box so that its center is at the predicted
        // location
        tracks[i].predPosition = Rect(Point2d(predictedCentroid.x - bbox.width / 2,
                                              predictedCentroid.y - bbox.height / 2),
                                      bbox.size());
    }
}

void App::calculateCostMatrix(vector<Track> &tracks, vector<Detection> &detections, vector<vector<double>> &costMatrix)
{
    // Iterate over each track-detection pair
    for (unsigned int i = 0; i < tracks.size(); i++)
    {
        costMatrix.push_back(vector<double>());

        Rect predBbox = tracks[i].predPosition;
        for (unsigned int j = 0; j < detections.size(); j++)
        {
            Rect bbox = detections[j].bbox;

            double overlap = computeBoundingBoxOverlap(predBbox, bbox);

            // Cost = 1 - overlap
            costMatrix[i].push_back(1.0 - overlap);
        }
    }
}

void App::classifyAssignments(vector<unsigned int> &assignmentPerRow, unsigned int numTracks, unsigned int numDetections,
                              vector<int> &assignments, vector<unsigned int> &unassignedTracks, vector<unsigned int> &unassignedDetections)
{
    std::vector<bool> isDetectionAssigned(numDetections, false);
    for (unsigned int i = 0; i < numTracks; i++)
    {
        if (assignmentPerRow[i] < numDetections)
        {
            assignments.push_back(assignmentPerRow[i]);

            isDetectionAssigned[assignmentPerRow[i]] = true;
        }
        else
        {
            assignments.push_back(-1);
            unassignedTracks.push_back(i);
        }
    }

    for (unsigned int i = 0; i < numDetections; i++)
    {
        if (!isDetectionAssigned[i])
        {
            unassignedDetections.push_back(i);
        }
    }
}

void App::solveAssignmentProblem(vector<vector<double>> &costMatrix, unsigned int numTracks, unsigned int numDetections,
                                 vector<int> &assignments, vector<unsigned int> &unassignedTracks, vector<unsigned int> &unassignedDetections,
                                 bool debug)
{
    double hugeNumber = 10000000.0;

    bool is_problematic = this->frame_id == 0;

    if (debug)
    {
        std::cout << "Frame number: " << this->frame_id << std::endl;

        if (is_problematic)
        {
            std::cout << "# tracks (rows): " << numTracks << std::endl;
            std::cout << "# detections (columns): " << numDetections << std::endl;

            // Print the cost matrix
            for (unsigned i = 0; i < numTracks; i++)
            {
                std::cout << "Row " << i << ": ";
                for (unsigned j = 0; j < numDetections; j++)
                {
                    if (j > 0)
                    {
                        std::cout << ",";
                    }
                    std::cout << costMatrix[i][j];
                }
                std::cout << std::endl;
            }
        }
    }

    // First, expand the cost matrix to be square (columns first; detections)
    if (numTracks > numDetections)
    {
        for (unsigned int i = 0; i < numTracks; i++)
        {
            for (unsigned int j = 0; j < numTracks - numDetections; j++)
            {
                costMatrix[i].push_back(args.costOfNonAssignment * 2);
            }
        }
    }

    if (debug && is_problematic)
    {
        std::cout << std::endl << "after expanding columns:" << std::endl;

        // Print the cost matrix
        for (unsigned i = 0; i < costMatrix.size(); i++)
        {
            std::cout << "Row " << i << ": ";
            for (unsigned j = 0; j < costMatrix[i].size(); j++)
            {
                if (j > 0)
                {
                    std::cout << ",";
                }
                std::cout << costMatrix[i][j];
            }
            std::cout << std::endl;
        }
    }

    // ... or expand the rows (tracks)
    if (numDetections > numTracks)
    {
        for (unsigned int i = 0; i < numDetections - numTracks; i++)
        {
            costMatrix.push_back(vector<double>());

            for (unsigned int j = 0; j < numDetections; j++)
            {
                costMatrix[numTracks + i].push_back(args.costOfNonAssignment * 2);
            }
        }
    }

    if (debug && is_problematic)
    {
        std::cout << std::endl << "after expanding rows, too:" << std::endl;

        // Print the cost matrix
        for (unsigned i = 0; i < costMatrix.size(); i++)
        {
            std::cout << "Row " << i << ": ";
            for (unsigned j = 0; j < costMatrix[i].size(); j++)
            {
                if (j > 0)
                {
                    std::cout << ",";
                }
                std::cout << costMatrix[i][j];
            }
            std::cout << std::endl;
        }
    }

    unsigned int n = costMatrix.size();
    std::vector<unsigned int> assignmentPerRow(n, n); // default to n for each index (invalid)

    if (n == 0)
    {
        App::classifyAssignments(assignmentPerRow, numTracks, numDetections, assignments, unassignedTracks, unassignedDetections);

        std::cout << "No assignment of detections to tracks to do." << std::endl;

        return;
    }

    // Step 1: subtract the row minima
    for (unsigned int rowIdx = 0; rowIdx < n; rowIdx++)
    {
        double minElt = hugeNumber;
        for (unsigned int colIdx = 0; colIdx < n; colIdx++)
        {
            minElt = (costMatrix[rowIdx][colIdx] < minElt) ? costMatrix[rowIdx][colIdx] : minElt;
        }

        bool found_zero = false;
        for (unsigned int colIdx = 0; colIdx < n; colIdx++)
        {
            costMatrix[rowIdx][colIdx] -= minElt;
            if (costMatrix[rowIdx][colIdx] == 0.0)
            {
                found_zero = true;
            }
        }

        if (!found_zero)
        {
            std::cout << "DID NOT ZERO OUT AN ELEMENT IN THE ROW!!!" << std::endl;
        }
    }

    if (debug && is_problematic)
    {
        std::cout << std::endl << "After step 1 (subtracting row minima):" << std::endl;

        // Print the cost matrix
        for (unsigned i = 0; i < costMatrix.size(); i++)
        {
            std::cout << "Row " << i << ": ";
            for (unsigned j = 0; j < costMatrix[i].size(); j++)
            {
                if (j > 0)
                {
                    std::cout << ",";
                }
                std::cout << costMatrix[i][j];
            }
            std::cout << std::endl;
        }
    }

    // Step 2: subtract the column minima
    for (unsigned int colIdx = 0; colIdx < n; colIdx++)
    {
        double minElt = hugeNumber;
        for (unsigned int rowIdx = 0; rowIdx < n; rowIdx++)
        {
            minElt = (costMatrix[rowIdx][colIdx] < minElt) ? costMatrix[rowIdx][colIdx] : minElt;
        }

        bool found_zero = false;
        for (unsigned int rowIdx = 0; rowIdx < n; rowIdx++)
        {
            costMatrix[rowIdx][colIdx] -= minElt;
            if (costMatrix[rowIdx][colIdx] == 0.0)
            {
                found_zero = true;
            }
        }

        if (!found_zero)
        {
            std::cout << "DID NOT ZERO OUT AN ELEMENT IN THE COLUMN!!!" << std::endl;
        }
    }

    if (debug && is_problematic)
    {
        std::cout << std::endl << "After step 2 (subtracting column minima):" << std::endl;

        // Print the cost matrix
        for (unsigned i = 0; i < costMatrix.size(); i++)
        {
            std::cout << "Row " << i << ": ";
            for (unsigned j = 0; j < costMatrix[i].size(); j++)
            {
                if (j > 0)
                {
                    std::cout << ",";
                }
                std::cout << costMatrix[i][j];
            }
            std::cout << std::endl;
        }
    }

    // Repeat until done
    int repeatCount = 0;
    while (true)
    {
        repeatCount++;
        if (debug && is_problematic && repeatCount < 5)
        {
            cout << endl << repeatCount << endl;
        }

        // Check for a completed assignment
        std::vector<bool> isRowAssigned(n, false);
        std::vector<bool> isColAssigned(n, false);

        unsigned int numAssigned = 0;
        assignmentPerRow.assign(n, n);

        // Assign as many rows as possible
        bool madeAssignment = true;
        while (madeAssignment)
        {
            madeAssignment = false;

            // First, make a pass over rows with only one available 0
            for (unsigned int rowIdx = 0; rowIdx < n; rowIdx++)
            {
                if (isRowAssigned[rowIdx]) { continue; }

                std::vector<unsigned int> zeroIdxs;
                for (unsigned int colIdx = 0; colIdx < n; colIdx++)
                {
                    if (App::equalsZero(costMatrix[rowIdx][colIdx]))// and !isColAssigned[colIdx])
                    {
                        zeroIdxs.push_back(colIdx);
                    }
                }

                //if (zeroIdxs.size() == 1)
                if (zeroIdxs.size() == 1 && !isColAssigned[zeroIdxs[0]])
                {
                    unsigned int colIdx = zeroIdxs[0];
                    isRowAssigned[rowIdx] = true;
                    isColAssigned[colIdx] = true;

                    assignmentPerRow[rowIdx] = colIdx;
                    madeAssignment = true;
                    numAssigned++;

                    if (debug && is_problematic && repeatCount < 5)
                    {
                        std::cout << "[row] Assigned row " << rowIdx << " to column " << colIdx << std::endl;
                    }
                }
            }

            // Next, make a pass over columns with only one available 0
            for (unsigned int colIdx = 0; colIdx < n; colIdx++)
            {
                if (isColAssigned[colIdx]) { continue; }

                std::vector<unsigned int> zeroIdxs;
                for (unsigned int rowIdx = 0; rowIdx < n; rowIdx++)
                {
                    if (App::equalsZero(costMatrix[rowIdx][colIdx]))// and !isRowAssigned[rowIdx])
                    {
                        zeroIdxs.push_back(rowIdx);
                    }
                }

                //if (zeroIdxs.size() == 1)
                if (zeroIdxs.size() == 1 && !isRowAssigned[zeroIdxs[0]])
                {
                    unsigned int rowIdx = zeroIdxs[0];
                    isRowAssigned[rowIdx] = true;
                    isColAssigned[colIdx] = true;

                    assignmentPerRow[rowIdx] = colIdx;
                    madeAssignment = true;
                    numAssigned++;

                    if (debug && is_problematic && repeatCount < 5)
                    {
                        std::cout << "[col] Assigned row " << rowIdx << " to column " << colIdx << std::endl;
                    }
                }
            }

            // Finally, if no assignments have been made yet this round, check
            // all rows and columns
            if (!madeAssignment)
            {
                if (debug && is_problematic && repeatCount < 5)
                {
                    std::cout << "No assignments this round, current assignments per row (" << numAssigned << " total): ";
                    for (unsigned row_idx = 0; row_idx < n; row_idx++)
                    {
                        if (row_idx > 0) { std::cout << ","; }
                        std::cout << assignmentPerRow[row_idx];
                    }
                    std::cout << std::endl;
                }

                for (unsigned int rowIdx = 0; rowIdx < n; rowIdx++)
                {
                    if (isRowAssigned[rowIdx]) { continue; }

                    std::vector<unsigned int> zeroIdxs;
                    for (unsigned int colIdx = 0; colIdx < n; colIdx++)
                    {
                        //if (costMatrix[rowIdx][colIdx] == 0.0 and !isColAssigned[colIdx])
                        if (App::equalsZero(costMatrix[rowIdx][colIdx]))
                        {
                            zeroIdxs.push_back(colIdx);
                        }
                    }

                    for (unsigned int zeroIdx = 0; zeroIdx < zeroIdxs.size(); zeroIdx++)
                    {
                        unsigned int colIdx = zeroIdxs[zeroIdx];
                        if (!isColAssigned[colIdx])
                        {
                            isRowAssigned[rowIdx] = true;
                            isColAssigned[colIdx] = true;

                            assignmentPerRow[rowIdx] = colIdx;
                            madeAssignment = true;
                            numAssigned++;
                            break;
                        }
                    }
                }
            }
        }

        // If all rows are assigned, we're done
        if (numAssigned == n)
        {
            break;
        }

        if (debug && is_problematic && repeatCount < 5)
        {
            std::cout << "Assignments per row (" << numAssigned << " total): ";
            for (unsigned row_idx = 0; row_idx < n; row_idx++)
            {
                if (row_idx > 0) { std::cout << ","; }
                std::cout << assignmentPerRow[row_idx];
            }
            std::cout << std::endl;
        }

        // Step 3: cover all zeros with a minimum number of "lines"
        std::vector<bool> isRowMarked(n, false);
        std::vector<bool> isColMarked(n, false);

        for (unsigned int rowIdx = 0; rowIdx < n; rowIdx++)
        {
            if (!isRowAssigned[rowIdx])
            {
                isRowMarked[rowIdx] = true;
            }
        }

        if (debug && is_problematic && repeatCount < 5)
        {
            std::cout << "Row markings: ";
            for (unsigned row_idx = 0; row_idx < n; row_idx++)
            {
                if (row_idx > 0) { std::cout << ","; }
                std::cout << isRowMarked[row_idx];
            }
            std::cout << std::endl;
        }

        while (true)
        {
            unsigned int newlyMarkedColCount = 0;

            for (unsigned int rowIdx = 0; rowIdx < n; rowIdx++)
            {
                if (!isRowMarked[rowIdx]) { continue; }

                for (unsigned int colIdx = 0; colIdx < n; colIdx++)
                {
                    if (App::equalsZero(costMatrix[rowIdx][colIdx]) &&
                        !isColMarked[colIdx])
                    {
                        isColMarked[colIdx] = true;
                        newlyMarkedColCount++;
                    }
                }
            }

            for (unsigned int colIdx = 0; colIdx < n; colIdx++)
            {
                if (!isColMarked[colIdx]) { continue; }

                for (unsigned int otherRowIdx = 0; otherRowIdx < n; otherRowIdx++)
                {
                    if (assignmentPerRow[otherRowIdx] == colIdx)
                    {
                        isRowMarked[otherRowIdx] = true;
                    }
                }
            }

            if (newlyMarkedColCount == 0)
            {
                break;
            }
        }

        if (debug && is_problematic && repeatCount < 5)
        {
            std::cout << "Column markings: ";
            for (unsigned colIdx = 0; colIdx < n; colIdx++)
            {
                if (colIdx > 0) { std::cout << ","; }
                std::cout << isColMarked[colIdx];
            }
            std::cout << std::endl;

            std::cout << "Updated row markings: ";
            for (unsigned row_idx = 0; row_idx < n; row_idx++)
            {
                if (row_idx > 0) { std::cout << ","; }
                std::cout << isRowMarked[row_idx];
            }
            std::cout << std::endl;
        }

        std::vector<bool> isRowCovered(n, false);
        bool allMarked = true;
        for (unsigned int rowIdx = 0; rowIdx < n; rowIdx++)
        {
            isRowCovered[rowIdx] = !isRowMarked[rowIdx];
            if (!isRowMarked[rowIdx])
            {
                allMarked = false;
            }
        }

        // If all rows are marked, then we are done
        if (allMarked)
        {
            break;
        }

        std::vector<bool> isColCovered(isColMarked);

        if (debug && is_problematic && repeatCount < 5)
        {
            std::cout << "Covered rows: ";
            for (unsigned row_idx = 0; row_idx < n; row_idx++)
            {
                if (row_idx > 0) { std::cout << ","; }
                std::cout << isRowCovered[row_idx];
            }
            std::cout << std::endl;

            std::cout << "Covered columns: ";
            for (unsigned colIdx = 0; colIdx < n; colIdx++)
            {
                if (colIdx > 0) { std::cout << ","; }
                std::cout << isColCovered[colIdx];
            }
            std::cout << std::endl;
        }

        // Step 4: create additional zeros
        double minUncoveredElt = hugeNumber;
        for (unsigned int rowIdx = 0; rowIdx < n; rowIdx++)
        {
            if (isRowCovered[rowIdx]) { continue; }

            for (unsigned int colIdx = 0; colIdx < n; colIdx++)
            {
                if (isColCovered[colIdx]) { continue; }

                minUncoveredElt = (costMatrix[rowIdx][colIdx] < minUncoveredElt) ? costMatrix[rowIdx][colIdx] : minUncoveredElt;
            }
        }

        if (debug && is_problematic && repeatCount < 5)
        {
            std::cout << "Minimum uncovered element: " << minUncoveredElt << std::endl;
        }

        // Subtract the minimum uncovered element from all uncovered elements,
        // and add it to all elements covered twice
        for (unsigned int rowIdx = 0; rowIdx < n; rowIdx++)
        {
            for (unsigned int colIdx = 0; colIdx < n; colIdx++)
            {
                if (!isRowCovered[rowIdx] && !isColCovered[colIdx])
                {
                    costMatrix[rowIdx][colIdx] -= minUncoveredElt;
                }
                else if (isRowCovered[rowIdx] && isColCovered[colIdx])
                {
                    costMatrix[rowIdx][colIdx] += minUncoveredElt;
                }
            }
        }
    }

    App::classifyAssignments(assignmentPerRow, numTracks, numDetections,
                             assignments, unassignedTracks, unassignedDetections);
}

void App::detectionToTrackAssignment(vector<Detection> &detections, vector<Track> &tracks, vector<int> &assignments, vector<unsigned int> &unassignedTracks, vector<unsigned int> &unassignedDetections, cv::Ptr<cv::cuda::HOG> gpu_hog, bool debug)
{
    // Compute the cost (based on overlap ratio) of assigning each detection to
    // each track - store results in #tracks x #detections matrix
    vector<vector<double>> costMatrix; // per track, then per detection
    App::calculateCostMatrix(tracks, detections, costMatrix);

    // TODO: add gating cost

    // Solve assignment, taking into account the cost of not assigning
    // any detections to a given track
    if (debug) { cout << "Solving assignment problem" << endl; }
    App::solveAssignmentProblem(costMatrix, tracks.size(), detections.size(),
                                assignments, unassignedTracks, unassignedDetections, debug);
    if (debug) { cout << "Done solving assignment problem" << endl; }
}

void App::updateTrackConfidence(Track *track)
{
    unsigned int numScoresToUse = (track->scores.size() < args.timeWindowSize) ? track->scores.size() : args.timeWindowSize;
    double maxScore = 0.0, sumScores = 0.0;
    for (unsigned int scoreIdx = track->scores.size() - numScoresToUse; scoreIdx < track->scores.size(); scoreIdx++)
    {
        double score = track->scores[scoreIdx];
        sumScores += score;

        if (score > maxScore)
        {
            maxScore = score;
        }
    }

    track->maxConfidence = maxScore;
    track->avgConfidence = sumScores / numScoresToUse;
}

/*
 * Updates the assigned tracks with the corresponding detections.
 */
void App::updateAssignedTracks(vector<Track> &tracks, vector<Detection> &detections, vector<double> &confidenceScores, vector<int> &assignments)
{
    for (unsigned int trackIdx = 0; trackIdx < tracks.size(); trackIdx++)
    {
        if (assignments[trackIdx] < 0) { continue; }

        unsigned int detectionIdx = assignments[trackIdx];

        Track *track = &(tracks[trackIdx]);
        Rect *detectionBbox = &(detections[detectionIdx]).bbox;

        // Stabilize the bounding box by taking the average of the size
        // of recent (up to) 4 boxes on the track
        unsigned int numPriorBboxes = (track->bboxes.size() < 4) ? track->bboxes.size() : 4;
        unsigned int w, h;

        unsigned int wsum = 0, hsum = 0;
        for (unsigned int bboxIdx = track->bboxes.size() - numPriorBboxes; bboxIdx < track->bboxes.size(); bboxIdx++)
        {
            wsum += track->bboxes[bboxIdx].width;
            hsum += track->bboxes[bboxIdx].height;
        }

        w = (wsum + detectionBbox->width) / (numPriorBboxes + 1);
        h = (hsum + detectionBbox->height) / (numPriorBboxes + 1);

        // Update the track with the bounding box
        Point2d centroid(detectionBbox->x, detectionBbox->y);
        centroid.x += (detectionBbox->width / 2) - (w / 2);
        centroid.y += (detectionBbox->height / 2) - (h / 2);
        track->bboxes.push_back(Rect(centroid, Size(w, h)));

        // Update the track's age, visibility, and score history
        track->age++;
        track->totalVisibleCount++;
//        track->scores.push_back(confidenceScores[detectionIdx]);

        // Update the track's confidence score based on the maximum detection
        // score in the past 'timeWindowSize' frames
        App::updateTrackConfidence(track);
    }
}

/*
 * Updates the tracks that were not assigned detections this frame.
 */
void App::updateUnassignedTracks(vector<Track> &tracks, vector<unsigned int> &unassignedTracks)
{
    for (unsigned int unassignedIdx = 0; unassignedIdx < unassignedTracks.size(); unassignedIdx++)
    {
        unsigned trackIdx = unassignedTracks[unassignedIdx];
        Track *track = &(tracks[trackIdx]);

        // Update the track's age, append the predicted bounding box, and
        // set the confidence to 0 (to indicate we don't know why it
        // did not get a detection)
        track->age++;
        track->bboxes.push_back(track->predPosition);
        track->scores.push_back(0.0);

        // Update the track's confidence based on the maximum detection
        // score in the past 'timeWindowSize' frames
        App::updateTrackConfidence(track);
    }
}

void App::deleteLostTracks(vector<Track> &tracks)
{
    vector<unsigned int> trackIndicesToDelete;

    for (unsigned int trackIdx = 0; trackIdx < tracks.size(); trackIdx++)
    {
        Track *track = &(tracks[trackIdx]);

        // Compute the fraction of the track's age for which it was visible
        double visibility = ((double)track->totalVisibleCount) / track->age;

        // Determine if the track is lost based on the visible fraction
        // and the confidence
        if ((track->age <= args.trackAgeThreshold && visibility <= args.trackVisibilityThreshold) ||
            (track->maxConfidence <= args.trackConfidenceThreshold))
        {
            trackIndicesToDelete.push_back(trackIdx);
        }
    }

    // Remove the lost tracks (maybe not the most efficient implementation)
    for (int i = trackIndicesToDelete.size() - 1; i >= 0; i--)
    {
        tracks.erase(tracks.begin() + trackIndicesToDelete[i]);
    }
}

/*
 * Creates new tracks from unassigned detections; assumes that any unassigned
 * detection is the start of a new track.
 */
void App::createNewTracks(vector<Track> &tracks, vector<Detection> &detections, vector<unsigned int> &unassignedDetections, vector<double> &confidenceScores)
{
    //cout << "Called createNewTracks with " << unassignedDetections.size() << " new detections." << endl;

    for (unsigned int unassignedIdx = 0; unassignedIdx < unassignedDetections.size(); unassignedIdx++)
    {
        unsigned detectionIdx = unassignedDetections[unassignedIdx];

        //Track newTrack(detections[detectionIdx], confidenceScores[detectionIdx]);
        Track newTrack(detections[detectionIdx].bbox, 0.0);

        tracks.push_back(newTrack);
    }
}

Track::Track(Rect &bbox, double score)
{
    this->id = nextTrackId++;

    int r = rand() % 256;
    int g = rand() % 256;
    int b = rand() % 256;
    this->color = Scalar(r, g, b);

    this->bboxes.push_back(bbox);
    this->scores.push_back(score);

    this->motionModel = &constantVelocityMotionModel;

    this->age = 1;
    this->totalVisibleCount = 1;

    this->maxConfidence = score;
    this->avgConfidence = score;

    this->predPosition = bbox;
}

Point2d constantVelocityMotionModel(Track &track)
{
    // If the track has only a single detection, just return the same position
    if (track.age == 1)
    {
        // Translate the top-left corner of the box to the center
        return Point2d(track.bboxes.back().x + track.bboxes.back().width / 2,
                       track.bboxes.back().y + track.bboxes.back().height / 2);
    }

    // Otherwise, assume constant velocity between the past two detections
    double dx, dy;
    dx = track.bboxes[track.bboxes.size()-1].x - track.bboxes[track.bboxes.size()-2].x;
    dy = track.bboxes[track.bboxes.size()-1].y - track.bboxes[track.bboxes.size()-2].y;
    return Point2d(track.bboxes.back().x + track.bboxes.back().width / 2 + dx,
                   track.bboxes.back().y + track.bboxes.back().height / 2 + dy);
}

double computeBoundingBoxOverlap(Rect &predBbox, Rect &bbox)
{
    // Compute the boundaries of the intersecting region
    double xleft = std::max(predBbox.tl().x, bbox.tl().x);
    double xright = std::min(predBbox.br().x, bbox.br().x);
    double ytop = std::max(predBbox.tl().y, bbox.tl().y);
    double ybottom = std::min(predBbox.br().y, bbox.br().y);

    // Check for no overlap
    if ((xright < xleft) || (ybottom < ytop))
    {
        return 0.0;
    }

    // Otherwise, compute the area of the intersection and union
    double intersectionArea = (xright - xleft) * (ybottom - ytop);
    double unionArea = predBbox.area() + bbox.area() - intersectionArea;

    // Finally, compute the overlap (in [0,1]) as intersection/union
    double overlap = intersectionArea / unionArea;
    return overlap;
}