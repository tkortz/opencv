#ifndef OPENCV_TBD_HPP
#define OPENCV_TBD_HPP

#ifndef __cplusplus
#  error tbd.hpp header must be compiled as C++
#endif

#include <map>
#include <opencv2/core/utility.hpp>

using namespace std;

/**
  @addtogroup cuda
  @{
    @defgroup tbd Tracking by Detection
  @}
 */

namespace cv { namespace tbd {

class Track;
class Tracker;

class CV_EXPORTS TbdArgs
{
public:
    TbdArgs(double costOfNonAssignment,
            unsigned int timeWindowSize,
            unsigned int trackAgeThreshold,
            double trackVisibilityThreshold,
            double trackConfidenceThreshold,
            bool shouldStoreMetrics);

    double costOfNonAssignment;
    unsigned int timeWindowSize;
    unsigned int trackAgeThreshold;
    double trackVisibilityThreshold;
    double trackConfidenceThreshold;
    bool shouldStoreMetrics;
};

/*
 * A trajectory corresponds to the ground-truth sequence of positions
 * of a tracked object.
 */
class CV_EXPORTS Trajectory
{
public:
    Trajectory(); // necessary to put it in a map
    Trajectory(int id);

    int id;

    vector<int> presentFrames;
    map<int, Rect> positionPerFrame;
    map<int, Vec3d> worldPositionPerFrame;

    // Tracking information per frame, assumed to exist for a given key (frame ID)
    // only if isTrackedPerFrame[key] == true
    map<int, bool> isTrackedPerFrame;
    map<int, int> trackIdPerFrame;
    map<int, Rect> predPosPerFrame;
    map<int, Rect> trackPosPerFrame;
    map<int, double> bboxOverlapPerFrame;

    int getFirstFrame();

    void addPosition(int frame, Rect &bbox, Vec3d &worldPos);
    void addTrackingInfo(int frame, Track *track);
};

/*
 * A detection corresponds to an observed (potentially incorrectly)
 * position of an object of interest.
 */
class CV_EXPORTS Detection
{
public:
    Detection(int id, int frame_id, Rect &bbox, double confidence);

    int id; // -1 if not known
    int frame_id;

    Rect bbox; // in 2D image coordinates

    Vec3d worldPosition; // in 3D world coordinates

    double confidence; // between 0.0 and 1.0; -1.0 if not calculated/known
};

/*
 * A track corresponds to a sequence of detections matched
 * together over time.
 */
class CV_EXPORTS Track
{
public:
    Track(Detection &detection, Tracker *tracker);
    Track(const Track &t); // copy constructor

    unsigned int id;

    Scalar color;

    std::vector<Vec3d> worldCoords;
    std::vector<Rect> bboxes;
    std::vector<double> scores;
    std::vector<int> frames; // a track might not be present in each frame (see history_distribution)

    Point2d (*motionModel)(Track &track, int frame_id);

    unsigned int age;
    unsigned int totalVisibleCount;

    double maxConfidence;
    double avgConfidence;

    Rect predPosition;
    double bboxOverlap;
};

/*
 * A tracker to track targets using tracking-by-detection.
 */
class CV_EXPORTS Tracker
{
public:
    Tracker(const TbdArgs *args);

    void reset();

    unsigned int getNextTrackId();

    void setTracks(vector<Track> &tracks);
    vector<Track>& getTracks();

    // Perform a single tracking step given a set of detections
    void performTrackingStep(vector<Detection> &foundDetections,
                             map<int, Trajectory> &trajectoryMap,
                             int frame_id);

    const TbdArgs *args;

    // Metrics
    vector<int> truePositives;  // TP_t: # assigned detections
    vector<int> falseNegatives; // FN_t: # unmatched detections
    vector<int> falsePositives; // FP_t: # unmatched tracks (hypotheses)
    vector<int> groundTruths;   // GT_t: # objects in the scene
    vector<int> numMatches;     // c_t:  # matches
    vector<double> bboxOverlap; // sum_i d_it: bbox overlap total for frame t

private:
    Tracker();

    unsigned int nextTrackId;

    vector<Track> tracks;

    // Predict new locations of tracks
    void predictNewLocationsOfTracks(int frame_id);
    void filterTracksOutOfBounds(int xmin, int xmax, int ymin, int ymax);

    // Match predictions to new detections (using Munkres' Algorithm)
    void detectionToTrackAssignment(vector<Detection> &detections, vector<int> &assignments, vector<unsigned int> &unassignedTracks, vector<unsigned int> &unassignedDetections, bool debug, int frameId);
    void calculateCostMatrix(vector<Detection> &detections, vector<vector<double>> &costMatrix);
    void classifyAssignments(vector<unsigned int> &assignmentPerRow, unsigned int numTracks, unsigned int numDetections,
                            vector<int> &assignments, vector<unsigned int> &unassignedTracks, vector<unsigned int> &unassignedDetections);
    void solveAssignmentProblem(vector<vector<double>> &costMatrix, unsigned int numTracks, unsigned int numDetections,
                                vector<int> &assignments, vector<unsigned int> &unassignedTracks, vector<unsigned int> &unassignedDetections,
                                bool debug, int frame_id);

    // Update tracks
    void updateAssignedTracks(vector<Detection> &detections, vector<int> &assignments);
    void updateUnassignedTracks(vector<unsigned int> &unassignedTracks, int frame_id);
    void deleteLostTracks();
    void createNewTracks(vector<Detection> &detections, vector<unsigned int> &unassignedDetections);
    void updateTrackConfidence(Track *track);

    inline bool equalsZero(double val)
    {
        return (val < 0.0) ? (val > -0.00000001) : (val < 0.00000001);
    }
};

}} // namespace cv { namespace tbd {

#endif /* OPENCV_TBD_HPP */