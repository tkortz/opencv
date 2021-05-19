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

#include <iostream>

namespace cv { namespace tbd {

double computeBoundingBoxOverlap(Rect &predBbox, Rect &bbox);
Point2d constantVelocityMotionModel(Track &track, int frame_id);

TbdArgs::TbdArgs()
{
    this->costOfNonAssignment = 10.0;
    this->timeWindowSize = 16;
    this->trackAgeThreshold = 4;
    this->trackVisibilityThreshold = 0.3;
    this->trackConfidenceThreshold = 0.2;
    this->shouldStoreMetrics = false;
}

TbdArgs::TbdArgs(double costOfNonAssignment,
                 int timeWindowSize,
                 int trackAgeThreshold,
                 double trackVisibilityThreshold,
                 double trackConfidenceThreshold,
                 bool shouldStoreMetrics)
{
    this->costOfNonAssignment = costOfNonAssignment;
    this->timeWindowSize = timeWindowSize;
    this->trackAgeThreshold = trackAgeThreshold;
    this->trackVisibilityThreshold = trackVisibilityThreshold;
    this->trackConfidenceThreshold = trackConfidenceThreshold;
    this->shouldStoreMetrics = shouldStoreMetrics;
}

Track::Track(Detection &detection, Tracker *tracker)
{
    this->id = tracker->getNextTrackId();

    int r = rand() % 256;
    int g = rand() % 256;
    int b = rand() % 256;
    this->color = Scalar(r, g, b);

    this->bboxes.push_back(detection.bbox);
    this->scores.push_back(detection.confidence);
    this->frames.push_back(detection.frame_id);
    this->worldCoords.push_back(detection.worldPosition);

    this->motionModel = &constantVelocityMotionModel;

    this->age = 1;
    this->totalVisibleCount = 1;

    this->maxConfidence = detection.confidence;
    this->avgConfidence = detection.confidence;

    this->predPosition = detection.bbox;
    this->bboxOverlap = 1.0;
}

Track::Track(const Track &t)
{
    this->id = t.id;

    this->color = t.color;

    for (unsigned didx = 0; didx < t.bboxes.size(); didx++)
    {
        this->bboxes.push_back(t.bboxes[didx]);
        this->scores.push_back(t.scores[didx]);
        this->frames.push_back(t.frames[didx]);
        this->worldCoords.push_back(t.worldCoords[didx]);
    }

    this->motionModel = t.motionModel;

    this->age = t.age;
    this->totalVisibleCount = t.totalVisibleCount;

    this->maxConfidence = t.maxConfidence;
    this->avgConfidence = t.avgConfidence;

    this->predPosition = t.predPosition;
    this->bboxOverlap = t.bboxOverlap;
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

void Trajectory::addPosition(int frame, Rect &bbox, Vec3d &worldPos)
{
    this->presentFrames.push_back(frame);
    this->positionPerFrame[frame] = bbox;
    this->worldPositionPerFrame[frame] = worldPos;
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

Detection::Detection()
{
    this->id = -1;
    this->frame_id = -1;
    this->bbox = Rect(0,0,0,0);
    this->confidence = -1;
}

Detection::Detection(int id, int frame_id, Rect &bbox, double confidence)
{
    this->id = id;
    this->frame_id = frame_id;
    this->bbox = bbox;
    this->confidence = confidence;
}

Tracker::Tracker(TbdArgs& _args)
{
    this->args = _args;
    this->nextTrackId = 0;
}

unsigned int Tracker::getNextTrackId()
{
    return this->nextTrackId++;
}

void Tracker::setTracks(vector<Track> &tracks)
{
    this->tracks = tracks;
}

vector<Track>& Tracker::getTracks()
{
    return this->tracks;
}

int Tracker::getNumTracks()
{
    return this->tracks.size();
}

Scalar Tracker::getTrackColorByIdx(int idx)
{
    return this->tracks.at(idx).color;
}

int Tracker::getTrackAgeByIdx(int idx)
{
    return this->tracks.at(idx).age;
}

double Tracker::getTrackMaxConfidenceByIdx(int idx)
{
    return this->tracks.at(idx).maxConfidence;
}

vector<Rect> Tracker::getTrackBboxesByIdx(int idx)
{
    return this->tracks.at(idx).bboxes;
}

vector<int> Tracker::getFrameNumsByIdx(int idx)
{
    return this->tracks.at(idx).frames;
}

void Tracker::reset()
{
    this->nextTrackId = 0;
    this->tracks.clear();

    this->truePositives.clear();
    this->falseNegatives.clear();
    this->falsePositives.clear();
    this->groundTruths.clear();
    this->numMatches.clear();
    this->bboxOverlap.clear();
}

void Tracker::prepDetectionForTrackingStep(Detection& detection, int frame_id)
{
    if (this->nextTrackingStepDetections.size() == 0)
    {
        this->nextTrackingStepFrameId = frame_id;
    }

    this->nextTrackingStepDetections.push_back(detection);
}

void Tracker::performTrackingStep(int frame_id)
{
    if (this->nextTrackingStepDetections.size() == 0 ||
        this->nextTrackingStepFrameId != frame_id)
    {
        return;
    }

    this->performTrackingStep(this->nextTrackingStepDetections, NULL, frame_id);

    this->nextTrackingStepDetections.clear();
    this->nextTrackingStepFrameId = -1;
}

void Tracker::performTrackingStep(vector<Detection> &foundDetections,
                                  std::map<int, Trajectory> *trajectoryMap,
                                  int frame_id)
{
    // Predict the new locations of the tracks
    this->predictNewLocationsOfTracks(frame_id);

    // Filter out tracks with predictions that are out of the window
    this->filterTracksOutOfBounds(0, 1280, 0, 720);

    // Use predicted track positions to map current detections to existing tracks
    std::vector<int> assignments;
    std::vector<unsigned int> unassignedTracks, unassignedDetections;
    this->detectionToTrackAssignment(foundDetections, assignments,
                                     unassignedTracks, unassignedDetections,
                                     false, frame_id);

    // Update the tracks (assigned and unassigned)
    this->updateAssignedTracks(foundDetections, assignments);
    this->updateUnassignedTracks(unassignedTracks, frame_id);

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
        if (!trajectoryMap || detection->id < 0) { continue; }

        Trajectory *trajectory = &((*trajectoryMap)[detection->id]);
        trajectory->addTrackingInfo(frame_id, track);
    }

    // Update trajectories for unassigned detections - note that this
    // will miss the first position/frame associated with each track,
    // because they are created just after
    for (unsigned udix = 0; udix < unassignedDetections.size(); udix++)
    {
        unsigned detectionIdx = unassignedDetections[udix];
        Detection *detection = &(foundDetections[detectionIdx]);

        // Ignore detections for which no ground truth information exists
        if (!trajectoryMap || detection->id < 0) { continue; }

        Trajectory *trajectory = &((*trajectoryMap)[detection->id]);
        trajectory->addTrackingInfo(frame_id, NULL);
    }

    // Delete any tracks that are lost enough, and create new tracks
    // for unassigned detections
    this->deleteLostTracks();
    this->createNewTracks(foundDetections, unassignedDetections);

    if (this->args.shouldStoreMetrics)
    {
        this->truePositives.push_back(numAssigned);
        this->falseNegatives.push_back(unassignedDetections.size());
        this->falsePositives.push_back(unassignedTracks.size());
        this->groundTruths.push_back(foundDetections.size());
        this->numMatches.push_back(numAssigned); // not necessarily the same as TP_t

        double bboxOverlap = 0.0;
        for (unsigned tidx = 0; tidx < tracks.size(); tidx++)
        {
            bboxOverlap += tracks[tidx].bboxOverlap;
        }
        this->bboxOverlap.push_back(bboxOverlap);
    }
}

void Tracker::predictNewLocationsOfTracks(int frame_id)
{
    for (unsigned int i = 0; i < tracks.size(); i++)
    {
        // Get the last bounding box on this track
        Rect &bbox = tracks[i].bboxes.back();

        // Predict the current location of the track
        Point2d predictedCentroid = tracks[i].motionModel(tracks[i], frame_id);

        // Shift the bounding box so that its center is at the predicted
        // location
        tracks[i].predPosition = Rect(Point2d(predictedCentroid.x - bbox.width / 2,
                                              predictedCentroid.y - bbox.height / 2),
                                      bbox.size());
    }
}

void Tracker::filterTracksOutOfBounds(int xmin, int xmax, int ymin, int ymax)
{
    std::vector<unsigned int> filteredTracks;
    for (unsigned i = 0; i < tracks.size(); i++)
    {
        Track &t = tracks[i];

        Rect &r = t.predPosition;

        // Check if the track's predicted position is in the window
        if (t.predPosition.br().x < xmin ||
            t.predPosition.tl().x >= xmax || // TAMERT HACK HACK HACK HACK
            t.predPosition.br().y < ymin ||
            t.predPosition.tl().y >= ymax)
        {
            filteredTracks.push_back(i);
        }
    }

    // Remove the filtered tracks (maybe not the most efficient implementation)
    for (int i = filteredTracks.size() - 1; i >= 0; i--)
    {
        cout << "Filtering out track that left scene with ID: " << tracks[filteredTracks[i]].id << endl;
        tracks.erase(tracks.begin() + filteredTracks[i]);
    }
}

void Tracker::calculateCostMatrix(vector<Detection> &detections, vector<vector<double>> &costMatrix)
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

void Tracker::classifyAssignments(vector<unsigned int> &assignmentPerRow, unsigned int numTracks, unsigned int numDetections,
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

void Tracker::solveAssignmentProblem(vector<vector<double>> &costMatrix, unsigned int numTracks, unsigned int numDetections,
                                     vector<int> &assignments, vector<unsigned int> &unassignedTracks, vector<unsigned int> &unassignedDetections,
                                     bool debug, int frameId)
{
    double hugeNumber = 10000000.0;

    bool is_problematic = frameId == 0;

    if (debug)
    {
        std::cout << "Frame number: " << frameId << std::endl;

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
                costMatrix[i].push_back(this->args.costOfNonAssignment * 2);
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
                costMatrix[numTracks + i].push_back(this->args.costOfNonAssignment * 2);
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
        this->classifyAssignments(assignmentPerRow, numTracks, numDetections, assignments, unassignedTracks, unassignedDetections);

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
                    if (this->equalsZero(costMatrix[rowIdx][colIdx]))// and !isColAssigned[colIdx])
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
                    if (this->equalsZero(costMatrix[rowIdx][colIdx]))// and !isRowAssigned[rowIdx])
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
                        if (this->equalsZero(costMatrix[rowIdx][colIdx]))
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
                    if (this->equalsZero(costMatrix[rowIdx][colIdx]) &&
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

    this->classifyAssignments(assignmentPerRow, numTracks, numDetections,
                              assignments, unassignedTracks, unassignedDetections);
}

void Tracker::detectionToTrackAssignment(vector<Detection> &detections, vector<int> &assignments, vector<unsigned int> &unassignedTracks, vector<unsigned int> &unassignedDetections, /*cv::Ptr<cv::cuda::HOG> gpu_hog,*/ bool debug, int frameId)
{
    // Compute the cost (based on overlap ratio) of assigning each detection to
    // each track - store results in #tracks x #detections matrix
    vector<vector<double>> costMatrix; // per track, then per detection
    this->calculateCostMatrix(detections, costMatrix);

    // TODO: add gating cost

    // Solve assignment, taking into account the cost of not assigning
    // any detections to a given track
    if (debug) { cout << "Solving assignment problem" << endl; }
    this->solveAssignmentProblem(costMatrix, tracks.size(), detections.size(),
                                assignments, unassignedTracks, unassignedDetections, debug, frameId);
    if (debug) {
        cout << "Done solving assignment problem: " << unassignedTracks.size() << " of " << tracks.size() << " tracks unassigned, ";
        cout << unassignedDetections.size() << " of " << detections.size() << " detections unassigned." << endl;
    }
}

void Tracker::updateTrackConfidence(Track *track)
{
    unsigned int numScoresToUse = (track->scores.size() < this->args.timeWindowSize) ? track->scores.size() : this->args.timeWindowSize;
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
void Tracker::updateAssignedTracks(vector<Detection> &detections, vector<int> &assignments)
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
        int w, h;

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
        track->frames.push_back(detections[detectionIdx].frame_id);
        track->worldCoords.push_back(detections[detectionIdx].worldPosition);

        // Update the metrics stored for the track
        track->bboxOverlap = computeBoundingBoxOverlap(*detectionBbox, track->predPosition);

        // Update the track's age, visibility, and score history
        track->age++;
        track->totalVisibleCount++;
        track->scores.push_back(detections[detectionIdx].confidence);

        // Update the track's confidence score based on the maximum detection
        // score in the past 'timeWindowSize' frames
        this->updateTrackConfidence(track);
    }
}

/*
 * Updates the tracks that were not assigned detections this frame.
 */
void Tracker::updateUnassignedTracks(vector<unsigned int> &unassignedTracks, int frame_id)
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
        track->frames.push_back(frame_id);
        track->worldCoords.push_back(Vec3d());

        // Update the stored metrics for the track
        track->bboxOverlap = 0.0;

        // Update the track's confidence based on the maximum detection
        // score in the past 'timeWindowSize' frames
        this->updateTrackConfidence(track);
    }
}

void Tracker::deleteLostTracks()
{
    vector<unsigned int> trackIndicesToDelete;

    for (unsigned int trackIdx = 0; trackIdx < tracks.size(); trackIdx++)
    {
        Track *track = &(tracks[trackIdx]);

        // Compute the fraction of the track's age for which it was visible
        double visibility = ((double)track->totalVisibleCount) / track->age;

        // Determine if the track is lost based on the visible fraction
        // and the confidence
        // TODO: remove check for maxConfidence >= when HOG gives confidence
        if ((track->age <= this->args.trackAgeThreshold && visibility <= this->args.trackVisibilityThreshold) ||
            (track->maxConfidence >= 0.0 && track->maxConfidence <= this->args.trackConfidenceThreshold))
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
void Tracker::createNewTracks(vector<Detection> &detections, vector<unsigned int> &unassignedDetections)
{
    //cout << "Called createNewTracks with " << unassignedDetections.size() << " new detections." << endl;

    for (unsigned int unassignedIdx = 0; unassignedIdx < unassignedDetections.size(); unassignedIdx++)
    {
        unsigned detectionIdx = unassignedDetections[unassignedIdx];

        Track newTrack(detections[detectionIdx], this);

        tracks.push_back(newTrack);
    }
}

Point2d constantVelocityMotionModel(Track &track, int frame_id)
{
    // If the track has only a single detection, just return the same position
    if (track.age == 1)
    {
        // Translate the top-left corner of the box to the center
        Rect &bPrev = track.bboxes.back();
        return Point2d(bPrev.x + bPrev.width / 2,
                       bPrev.y + bPrev.height / 2);
    }

    // Otherwise, assume constant velocity between the past two detections
    // (handle the case where the past two frame difference are not equal)
    double dx, dy;
    int fPrev1 = track.frames[track.frames.size() - 1];
    int fPrev2 = track.frames[track.frames.size() - 2];
    Rect &bPrev1 = track.bboxes[track.bboxes.size() - 1];
    Rect &bPrev2 = track.bboxes[track.bboxes.size() - 2];
    // std::cout << "fnow=" << frame_id << ", f1=" << fPrev1 << ", f2=" << fPrev2 << std::endl;
    double frameDifferenceRatio = ((double)(frame_id - fPrev1)) / (fPrev1 - fPrev2);
    dx = frameDifferenceRatio * (bPrev1.x - bPrev2.x);
    dy = frameDifferenceRatio * (bPrev1.y - bPrev2.y);
    double w = (bPrev1.width + bPrev2.width) / 2.0;
    double h = (bPrev1.height + bPrev2.height) / 2.0;
    return Point2d(bPrev1.x + w / 2 + dx,
                   bPrev1.y + h / 2 + dy);
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

}}