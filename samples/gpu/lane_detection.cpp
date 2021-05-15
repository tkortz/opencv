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
#include <opencv2/core/utility.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/cudaimgproc.hpp"

using namespace std;
using namespace cv;

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
};

class App
{
public:
    App(const Args& s);
    void run();

    void readFrames();
    void readImage(Mat& dst);
    void houghlines(Mat& src, Mat& img, int frame_num);

private:
    App operator=(App&);

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
         << " --count <number of frames to process if sequence (repeating if necessary)>\n";
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
         << endl;

    use_gpu = true;
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
        dst = imread(filenames[img_read_count++]);	// 0 --> .gitignore
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

void App::houghlines(Mat& src, Mat& img, int frame_num)
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
        cv::cuda::GpuMat d_src(mask);
        cv::cuda::GpuMat d_lines;

        const int64 start = getTickCount();

        Ptr<cv::cuda::HoughSegmentDetector> hough = cv::cuda::createHoughSegmentDetector(1.0f, (float) (CV_PI / 180.0f), 30, 200);

        hough->detect(d_src, d_lines);

        timeSec = (getTickCount() - start) / getTickFrequency();

        if (!d_lines.empty())
        {
            lines.resize(d_lines.cols);
            Mat h_lines(1, d_lines.cols, CV_32SC4, &lines[0]);
            d_lines.download(h_lines);
        }
    }

    // Display the detected lanes
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

void App::run()
{
    Mat frame, img, img_aux, img_thresholded;

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
        houghlines(frame, img_thresholded, count_frame % frames.size());

        handleKey((char)waitKey(3));

        count_frame++;
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