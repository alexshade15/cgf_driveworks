/////////////////////////////////////////////////////////////////////////////////////////
//
// Notice
// ALL NVIDIA DESIGN SPECIFICATIONS AND CODE ("MATERIALS") ARE PROVIDED "AS IS"
// NVIDIA MAKES NO REPRESENTATIONS, WARRANTIES, EXPRESSED, IMPLIED, STATUTORY,
// OR OTHERWISE WITH RESPECT TO THE MATERIALS, AND EXPRESSLY DISCLAIMS ANY
// IMPLIED WARRANTIES OF NONINFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
// PARTICULAR PURPOSE.
//
// NVIDIA CORPORATION & AFFILIATES assumes no responsibility for the
// consequences of use of such information or for any infringement of patents
// or other rights of third parties that may result from its use. No license is
// granted by implication or otherwise under any patent or patent rights of
// NVIDIA CORPORATION & AFFILIATES. No third party distribution is allowed
// unless expressly authorized by NVIDIA. Details are subject to change without
// notice. This code supersedes and replaces all information previously
// supplied. NVIDIA CORPORATION & AFFILIATES products are not authorized for
// use as critical components in life support devices or systems without
// express written approval of NVIDIA CORPORATION & AFFILIATES.
//
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.
//
/////////////////////////////////////////////////////////////////////////////////////////

#ifndef DW_FRAMEWORK_DETECTANDTRACKIMPL_HPP_
#define DW_FRAMEWORK_DETECTANDTRACKIMPL_HPP_

// Core
#include <dw/core/base/Version.h>
#include <dw/core/context/Context.h>

// DNN
#include <dw/dnn/DNN.h>

// Tracker
#include <dw/imageprocessing/featuredetector/FeatureDetector.h>
#include <dw/imageprocessing/tracking/boxtracker2d/BoxTracker2D.h>
#include <dw/imageprocessing/tracking/featuretracker/FeatureTracker.h>

// Image resize
#include <dw/imageprocessing/geometry/imagetransformation/ImageTransformation.h>

#include <dwcgf/node/SimpleNodeT.hpp>

#include "./detectAndTrack.hpp"

namespace dw {
namespace framework {

class detectAndTrackImpl : public dw::framework::SimpleNodeT<detectAndTrack> {
 public:
  static constexpr char LOG_TAG[] = "detectAndTrack";

  // Initialization and destruction
  detectAndTrackImpl(const dwContextHandle_t ctx);
  ~detectAndTrackImpl() override;

  void initInputPorts();
  void initOutputPorts();
  void registerPasses();

  // My methods
  void initialize();
  void reset_node();

 private:
  // Pass methods
  dwStatus processPass();

  // Internal states of node
  size_t m_epochCount{0};

  dwContextHandle_t m_ctx{DW_NULL_HANDLE};

  // ------------------------------------------------
  // Driveworks Context and SAL
  // ------------------------------------------------
  dwContextHandle_t m_sdk = DW_NULL_HANDLE;

  // ------------------------------------------------
  // DNN
  // ------------------------------------------------
  typedef std::pair<dwRectf, float32_t> BBoxConf;
  static constexpr float32_t COVERAGE_THRESHOLD = 0.6f;
  const uint32_t m_maxDetections = 1000U;
  const float32_t m_nonMaxSuppressionOverlapThreshold = 0.5;

  dwDNNHandle_t m_dnn = DW_NULL_HANDLE;
  dwDataConditionerHandle_t m_dataConditioner = DW_NULL_HANDLE;
  float32_t *m_dnnInputDevice = nullptr;
  float32_t *m_dnnOutputsDevice[2] = {nullptr};
  std::unique_ptr<float32_t[]> m_dnnOutputsHost[2] = {nullptr};

  uint32_t m_cvgIdx;
  uint32_t m_bboxIdx;
  dwBlobSize m_networkInputDimensions;
  dwBlobSize m_networkOutputDimensions[2];

  uint32_t m_totalSizeInput;
  uint32_t m_totalSizesOutput[2];
  dwRect m_detectionRegion;

  // ------------------------------------------------
  // YOLO: 1 input and 1 output
  // Switch to Yolo onnx from now on
  // ------------------------------------------------
  static constexpr float32_t CONFIDENCE_THRESHOLD = 0.45f;
  static constexpr float32_t SCORE_THRESHOLD = 0.25f;

  const std::string YOLO_CLASS_NAMES[80] = {
      "person",        "bicycle",      "car",
      "motorcycle",    "airplane",     "bus",
      "train",         "truck",        "boat",
      "traffic light", "fire hydrant", "stop sign",
      "parking meter", "bench",        "bird",
      "cat",           "dog",          "horse",
      "sheep",         "cow",          "elephant",
      "bear",          "zebra",        "giraffe",
      "backpack",      "umbrella",     "handbag",
      "tie",           "suitcase",     "frisbee",
      "skis",          "snowboard",    "sports ball",
      "kite",          "baseball bat", "baseball glove",
      "skateboard",    "surfboard",    "tennis racket",
      "bottle",        "wine glass",   "cup",
      "fork",          "knife",        "spoon",
      "bowl",          "banana",       "apple",
      "sandwich",      "orange",       "broccoli",
      "carrot",        "hot dog",      "pizza",
      "donut",         "cake",         "chair",
      "couch",         "potted plant", "bed",
      "dining table",  "toilet",       "tv",
      "laptop",        "mouse",        "remote",
      "keyboard",      "cell phone",   "microwave",
      "oven",          "toaster",      "sink",
      "refrigerator",  "book",         "clock",
      "vase",          "scissors",     "teddy bear",
      "hair drier",    "toothbrush"};

  std::vector<std::string> m_label;
  std::vector<dwTrackedBox2D> m_detectedTrackBox;

  // ------------------------------------------------
  // Feature Tracker
  // ------------------------------------------------
  uint32_t m_maxFeatureCount;
  uint32_t m_historyCapacity;

  dwFeature2DDetectorHandle_t m_featureDetector = DW_NULL_HANDLE;
  dwFeature2DTrackerHandle_t m_featureTracker = DW_NULL_HANDLE;

  dwFeatureHistoryArray m_featureHistoryCPU = {};
  dwFeatureHistoryArray m_featureHistoryGPU = {};
  dwFeatureArray m_featureDetectedGPU = {};

  dwPyramidImage m_pyramidPrevious = {};
  dwPyramidImage m_pyramidCurrent = {};

  dwImageTransformationHandle_t ImageTransformationEngine_ = DW_NULL_HANDLE;

  uint8_t *m_featureMask;
  size_t m_maskPitch;
  dwVector2ui m_maskSize{};

  // ------------------------------------------------
  // Box Tracker
  // ------------------------------------------------
  dwBoxTracker2DHandle_t m_boxTracker;
  std::vector<float32_t> m_previousFeatureLocations;
  std::vector<float32_t> m_currentFeatureLocations;
  std::vector<dwFeature2DStatus> m_featureStatuses;
  const dwTrackedBox2D *m_trackedBoxes = nullptr;
  size_t m_numTrackedBoxes = 0;
  std::vector<dwRectf> m_trackedBoxListFloat;

  dwImageHandle_t m_imageRGBA;
  cudaStream_t m_cudaStream = 0;
  uint32_t m_imageWidth;
  uint32_t m_imageHeight;

  // ------------------------------------------------
  // Methods
  // ------------------------------------------------
  float32_t calculateIouOfBoxes(dwRectf, dwRectf);
  YoloScoreRectVector nonMaximumSuppression(YoloScoreRectVector &, float32_t);
  void interpretOutput(const float32_t *, const dwRect *const);
  float32_t overlap(const dwRectf &, const dwRectf &);
  void runTracker(const dwImageCUDA *);
  uint32_t trackFeatures(const dwImageCUDA *);
  uint32_t updateFeatureLocationsStatuses();

  static constexpr size_t m_YoloScoreRectArraySize{512};
  YoloScoreRectArray mYoloScoreRectArray{};
  std::array<YoloScoreRect, m_YoloScoreRectArraySize> m_yoloData{0};
};

}  // namespace framework
}  // namespace dw

#endif  // DW_FRAMEWORK_DETECTANDTRACKIMPL_HPP_
