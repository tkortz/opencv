#ifndef OPENCV_TBD_HPP
#define OPENCV_TBD_HPP

#ifndef __cplusplus
#  error tbd.hpp header must be compiled as C++
#endif

#include <map>
#include <opencv2/core/utility.hpp>

using namespace std;

/**
  @defgroup tbd Tracking-By-Detection
*/

namespace cv
{

//! @addtogroup tbd
//! @{

namespace tbd {

class Track;
class Tracker;

class CV_EXPORTS_W_SIMPLE TbdArgs
{
public:
    CV_WRAP TbdArgs(double costOfNonAssignment,
                    int timeWindowSize,
                    int trackAgeThreshold,
                    double trackVisibilityThreshold,
                    double trackConfidenceThreshold,
                    bool shouldStoreMetrics);

    CV_PROP_RW double costOfNonAssignment;
    CV_PROP_RW int timeWindowSize;
    CV_PROP_RW int trackAgeThreshold;
    CV_PROP_RW double trackVisibilityThreshold;
    CV_PROP_RW double trackConfidenceThreshold;
    CV_PROP_RW bool shouldStoreMetrics;

    TbdArgs();
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
class CV_EXPORTS_W_SIMPLE Detection
{
public:
    CV_WRAP Detection();
    CV_WRAP Detection(int id, int frame_id, Rect &bbox, double confidence);

    CV_PROP_RW int id; // -1 if not known
    CV_PROP_RW int frame_id;

    CV_PROP_RW Rect bbox; // in 2D image coordinates

    CV_PROP_RW Vec3d worldPosition; // in 3D world coordinates

    CV_PROP_RW double confidence; // between 0.0 and 1.0; -1.0 if not calculated/known
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
class CV_EXPORTS_W Tracker
{
public:
    CV_WRAP Tracker(TbdArgs& _args);

    CV_WRAP void reset();

    unsigned int getNextTrackId();

    void setTracks(vector<Track> &tracks); // used for faking history choice
    vector<Track>& getTracks();

    // Somewhat hacky approach to Python API; assumes not ground-truth detection
    // (avoiding passing vector<Detection>)
    CV_WRAP void prepDetectionForTrackingStep(Detection& detection, int frame_id);
    CV_WRAP void performTrackingStep(int frame_id);

    // Perform a single tracking step given a set of detections
    void performTrackingStep(vector<Detection> &foundDetections,
                             map<int, Trajectory> *trajectoryMap,
                             int frame_id);

    // Python API
    CV_WRAP int getNumTracks();
    CV_WRAP Scalar getTrackColorByIdx(int idx);
    CV_WRAP int getTrackAgeByIdx(int idx);
    CV_WRAP double getTrackMaxConfidenceByIdx(int idx);
    CV_WRAP vector<Rect> getTrackBboxesByIdx(int idx);
    CV_WRAP vector<int> getFrameNumsByIdx(int idx);

    TbdArgs args;

    // Metrics (expose all to Python)
    CV_PROP vector<int> truePositives;  // TP_t: # assigned detections
    CV_PROP vector<int> falseNegatives; // FN_t: # unmatched detections
    CV_PROP vector<int> falsePositives; // FP_t: # unmatched tracks (hypotheses)
    CV_PROP vector<int> groundTruths;   // GT_t: # objects in the scene
    CV_PROP vector<int> numMatches;     // c_t:  # matches
    CV_PROP vector<double> bboxOverlap; // sum_i d_it: bbox overlap total for frame t

private:
    Tracker();

    unsigned int nextTrackId;

    vector<Track> tracks;

    // Things to make Python API work (prep one Detection at a time for next tracking step)
    vector<Detection> nextTrackingStepDetections;
    int nextTrackingStepFrameId;

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

//! @} tbd

} // namespace tbd {

} // namespace cv {

#endif /* OPENCV_TBD_HPP */