#!/usr/bin/python

"""
This example illustrates how to use Tracking-by-Detection.

Usage:
    # python3 tbd.py \\
    #         rgb_dir \\
    #         pedestrian_bbox_file.txt vehicle_bbox_file.txt \\
    #         pedestrian_output.txt vehicle_output.txt \\
    #         num_frames
"""

# Python 2/3 compatibility (although we assume Python 3)
from __future__ import print_function

import cv2 as cv
import numpy as np

import glob
import math
import sys

def parseBboxFile(filepath, perFrameBboxes, perFrameCameraPoses, numFrames):
    perFrameBboxes.clear()
    perFrameCameraPoses.clear()

    prevFrame = -1
    startFrame = -1
    currentFrameBboxes = []

    f = open(filepath, 'r')
    for line in f:
        if "|" not in line: continue

        s = line.strip().split('|')

        frameNum = int(s[0])
        assert len(s) == 5 # Only handle real detection data

        # Check if this is the first line read (assume they're in order by frame)
        if startFrame == -1:
            # Use the frame number provided
            startFrame = 0
            prevFrame = 0

        # If this is the first line we've seen from this frame, start a new vector
        # (there might have been multiple frames without any detections)
        for i in range(prevFrame, frameNum):
            perFrameBboxes.append(currentFrameBboxes)
            currentFrameBboxes = []

        # We're not using ground-truth detections, so we don't know the obj ID
        objId = 2

        # Get the info from this bounding box into a list
        bboxInfo = [objId] + [int(val) for val in s[1:]]

        currentFrameBboxes.append(bboxInfo)

        prevFrame = frameNum

    # Add the last frame's bounding boxes (and any that were missed)
    for i in range(prevFrame, startFrame + numFrames):
        perFrameBboxes.append(currentFrameBboxes)
        currentFrameBboxes = []

    f.close()

def parseDetections(perFrameBboxes, frameId):
    if len(perFrameBboxes) == 0:
        return []

    detections = []

    for bbox in perFrameBboxes[frameId]:
        objectId = bbox[0]
        r = (bbox[1], bbox[3], bbox[2] - bbox[1], bbox[4] - bbox[3]) # x, y, w, h

        d = cv.tbd_Detection()
        d.id = objectId
        d.frame_id = frameId
        d.bbox = r
        d.confidence = 1.0
        detections.append(d)

    return detections

def writeTrackingOutputToFile(tracker, filepath, frameId):
    # Open the file
    f = open(filepath, 'w')
    printDst = f

    # Write selected history ages
    print("history|1", file=printDst)

    # # Dervice trajectory-based metrics
    idSwapsPerFrame = [0] * frameId
    numMostlyTracked = 0
    numPartiallyTracked = 0
    numMostlyLost = 0

    # Write per-frame tracking evaluation metrics to the output file
    totalBboxOverlap = 0.0
    for fnum in range(len(tracker.truePositives)):
        # Write the frame number
        print("frame|" + str(fnum) + "|", end="", file=printDst)

        # Write the metrics for the frame
        print("TP," + str(tracker.truePositives[fnum][0]) + ";", end="", file=printDst)
        print("FN," + str(tracker.falseNegatives[fnum][0]) + ";", end="", file=printDst)
        print("FP," + str(tracker.falsePositives[fnum][0]) + ";", end="", file=printDst)
        print("GT," + str(tracker.groundTruths[fnum][0]) + ";", end="", file=printDst)
        print("c," + str(tracker.numMatches[fnum][0]) + ";", end="", file=printDst)
        print("IDSW," + str(idSwapsPerFrame[fnum]) + ";", end="", file=printDst)
        print("sum_di," + str.format("{0:0.5f}", tracker.bboxOverlap[fnum][0]), file=printDst)

        totalBboxOverlap += tracker.bboxOverlap[fnum][0]

    # Compute MOTA, A-MOTA, and MOTP for the scenario
    motaNumerator = 0.0
    amotaNumerator = 0.0
    motaDenominator = 0.0
    motpNumerator = totalBboxOverlap
    motpDenominator = 0.0
    for fnum in range(len(tracker.truePositives)):
        motaNumerator += tracker.falseNegatives[fnum] + tracker.falsePositives[fnum] + idSwapsPerFrame[fnum]
        amotaNumerator += tracker.falseNegatives[fnum] + tracker.falsePositives[fnum]
        motaDenominator += tracker.groundTruths[fnum]
        motpDenominator += tracker.numMatches[fnum]

    mota = (1 - (motaNumerator / motaDenominator))[0] if motaDenominator != 0 else "nan"
    amota = (1 - (amotaNumerator / motaDenominator))[0] if motaDenominator != 0 else "nan"
    motp = (motpNumerator / motpDenominator)[0] if motpDenominator != 0 else "nan"

    motaStr = str.format("{0:0.6f}", mota) if mota != "nan" else mota
    amotaStr = str.format("{0:0.6f}", amota) if amota != "nan" else amota
    motpStr = str.format("{0:0.6f}", motp) if motp != "nan" else motp

    # Write total scenario tracking evaluation metrics (MT, ML, MOTA, MOTP)
    print("scenario|", end="", file=printDst)
    print("MT," + str(numMostlyTracked) + ";", end="", file=printDst)
    print("PT," + str(numPartiallyTracked) + ";", end="", file=printDst)
    print("ML," + str(numMostlyLost) + ";", end="", file=printDst)
    print("MOTA," + motaStr + ";", end="", file=printDst)
    print("A-MOTA," + amotaStr + ";", end="", file=printDst)
    print("MOTP," + motpStr, file=printDst)

    # Close the output file
    f.close()

def drawRect(frame, r, color, thickness=3):
    # r is x, y, w, h
    tl = (r[0], r[1])
    br = (r[0]+r[2], r[1]+r[3])
    cv.rectangle(frame, tl, br, color, thickness)

def main():
    if len(sys.argv) < 7:
        raise Exception("Missing parameters, should have 7.")

    carla_img_folder = sys.argv[1]                 # folder of RGB images
    pedestrian_bbox_filepath = sys.argv[2]         # results from detector
    vehicle_bbox_filepath = sys.argv[3]            # results from detector
    pedestrian_tracking_out_filepath = sys.argv[4] # where to put tracking results
    vehicle_tracking_out_filepath = sys.argv[5]    # where to put tracking results
    num_tracking_frames = int(sys.argv[6])         # how many frames to track for

    # List of per-frame per-pedestrian/per-vehicle bboxes
    pedestrianPerFrameBboxes = []
    vehiclePerFrameBboxes = []

    # List of per-frame camera poses
    perFrameCameraPoses = []

    # Parse the pedestrian/vehicle bbox file
    parseBboxFile(pedestrian_bbox_filepath, pedestrianPerFrameBboxes, perFrameCameraPoses, num_tracking_frames)
    parseBboxFile(vehicle_bbox_filepath, vehiclePerFrameBboxes, perFrameCameraPoses, num_tracking_frames)

    # Build the trackers
    tbdArgs = cv.tbd_TbdArgs(10.0, 16, 4, 0.3, 0.2, True)
    pedestrianTracker = cv.tbd_Tracker(tbdArgs)
    vehicleTracker = cv.tbd_Tracker(tbdArgs)

    # Read in the first image
    filenames = glob.glob(carla_img_folder + "/*.*")
    filenames.sort()

    # Loop until we run out of stuff to do
    frameId = 0
    running = True
    while running and (frameId < num_tracking_frames):
        # Grab the image frame
        frame = cv.imread(filenames[frameId % len(filenames)])
        print("Loaded frame {0}".format(frameId))

        # Grab the detections for this frame
        pedestrianDetections = parseDetections(pedestrianPerFrameBboxes, frameId % len(filenames))
        vehicleDetections = parseDetections(vehiclePerFrameBboxes, frameId % len(filenames))

        # Reset the tracks at the beginning of the input
        if frameId == 0:
            pedestrianTracker.reset()
            vehicleTracker.reset()

        # Do the tracking step (hacky for now, but should work)
        for detection in pedestrianDetections:
            pedestrianTracker.prepDetectionForTrackingStep(detection, frameId)
        for detection in vehicleDetections:
            vehicleTracker.prepDetectionForTrackingStep(detection, frameId)
        pedestrianTracker.performTrackingStep(frameId)
        vehicleTracker.performTrackingStep(frameId)

        # Draw the detections
        for detection in pedestrianDetections:
            drawRect(frame, detection.bbox, (255, 0, 0))
        for detection in vehicleDetections:
            drawRect(frame, detection.bbox, (0, 255, 0))

        # Draw the tracks
        for tracker in [pedestrianTracker, vehicleTracker]:
            for trackIdx in range(tracker.getNumTracks()):
                # Don't draw tracks that are too new and/or with too low confidence
                trackAge = tracker.getTrackAgeByIdx(trackIdx)
                trackMaxConf = tracker.getTrackMaxConfidenceByIdx(trackIdx)
                if (trackAge < tbdArgs.trackAgeThreshold and trackMaxConf >= 0.0 and trackMaxConf < tbdArgs.trackConfidenceThreshold) or \
                   (trackAge < tbdArgs.trackAgeThreshold / 2.0):
                    continue

                # Draw the centroids of prior positions, using the bounding box centers
                # directly in screen space
                trackBboxes = tracker.getTrackBboxesByIdx(trackIdx)
                trackFrames = tracker.getFrameNumsByIdx(trackIdx)
                trackColor = tracker.getTrackColorByIdx(trackIdx)
                for bboxIdx in range(len(trackBboxes)-1):
                    # Choose a color for the centroid
                    bboxFrameNum = trackFrames[bboxIdx]
                    centroidAlpha = int(float(bboxFrameNum) / frameId * 255)
                    color = (trackColor[0], trackColor[1], trackColor[2], centroidAlpha)

                    # Draw the centroid point
                    bbox = trackBboxes[bboxIdx]
                    x, y, w, h = bbox
                    centroid = (x + int(w/2), y + int(h/2))
                    thickness = -1 # fill the circle
                    cv.circle(frame, centroid, 2, color, thickness)

                # Draw the rectangle for this frame
                drawRect(frame, trackBboxes[-1], (trackColor[0], trackColor[1], trackColor[2]))

        # Display info on the screen
        cv.putText(frame, "Frame: " + str(frameId), (5,25), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255,100,0), 2)
        cv.imshow("Tracking-by-Detection", frame)

        # Check for any key presses
        if cv.waitKey(3) & 0xFF == 27: # escape
           running = False

        frameId += 1

    # Write tracking metrics to a file
    writeTrackingOutputToFile(pedestrianTracker, pedestrian_tracking_out_filepath, frameId)
    writeTrackingOutputToFile(vehicleTracker, vehicle_tracking_out_filepath, frameId)


def test():
    args = cv.tbd_TbdArgs(10.0, 16, 4, 0.3, 0.2, False)
    tr = cv.tbd_Tracker(args)

    frameId = 7
    r = (100, 200, 30, 40) # x, y, w, h
    confidence = 0.5
    d = cv.tbd_Detection(-1, frameId, r, confidence)

    tr.prepDetectionForTrackingStep(d, frameId)
    tr.performTrackingStep(frameId)

    numTracks = tr.getNumTracks()
    for tid in range(numTracks):
        print(tr.getTrackColorByIdx(tid))
        print(tr.getTrackAgeByIdx(tid))
        print(tr.getTrackMaxConfidenceByIdx(tid))
        for bbox in tr.getTrackBboxesByIdx(tid):
            print(bbox)
        for frame_nums in tr.getFrameNumsByIdx(tid):
            print(frame_nums)


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()