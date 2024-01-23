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

#include "./detectAndTrackImpl.hpp"

namespace dw {
namespace framework {

bool sort_score(YoloScoreRect box1, YoloScoreRect box2) {
  return box1.score > box2.score ? true : false;
}

constexpr char detectAndTrackImpl::LOG_TAG[];

detectAndTrackImpl::detectAndTrackImpl(const dwContextHandle_t ctx) {
  DW_LOGD << "KINELOG - D&T - initInputPorts: " << __LINE__ << Logger::State::endl;
  initInputPorts();
  DW_LOGD << "KINELOG - D&T - initOutputPorts: " << __LINE__ << Logger::State::endl;
  initOutputPorts();
  DW_LOGD << "KINELOG - D&T - registerPasses: " << __LINE__ << Logger::State::endl;
  registerPasses();

  DW_LOGD << "KINELOG - D&T - initialize: " << __LINE__ << Logger::State::endl;
  initialize();
  DW_LOGD << "KINELOG - D&T - initializeation completed: " << __LINE__ << Logger::State::endl;
}

detectAndTrackImpl::~detectAndTrackImpl() {
  // Free GPU memory
  if (m_dnnOutputsDevice[0]) {
    CHECK_CUDA_ERROR(cudaFree(m_dnnOutputsDevice[0]));
  }
  if (m_featureMask) {
    CHECK_CUDA_ERROR(cudaFree(m_featureMask));
  }
  if (m_imageRGBA) {
    CHECK_DW_ERROR(dwImage_destroy(m_imageRGBA));
  }

  // Release box tracker
  CHECK_DW_ERROR(dwBoxTracker2D_release(m_boxTracker));

  // Release feature tracker and list
  CHECK_DW_ERROR(dwFeatureHistoryArray_destroy(m_featureHistoryCPU));
  CHECK_DW_ERROR(dwFeatureHistoryArray_destroy(m_featureHistoryGPU));
  CHECK_DW_ERROR(dwFeatureArray_destroy(m_featureDetectedGPU));
  CHECK_DW_ERROR(dwFeature2DDetector_release(m_featureDetector));
  CHECK_DW_ERROR(dwFeature2DTracker_release(m_featureTracker));

  // Release pyramids
  CHECK_DW_ERROR(dwPyramid_destroy(m_pyramidCurrent));
  CHECK_DW_ERROR(dwPyramid_destroy(m_pyramidPrevious));

  // Release ImageTransformationEngine
  CHECK_DW_ERROR(dwImageTransformation_release(ImageTransformationEngine_));

  // Release detector
  CHECK_DW_ERROR(dwDNN_release(m_dnn));
  // Release data conditioner
  CHECK_DW_ERROR(dwDataConditioner_release(m_dataConditioner));

  // Release SDK
  CHECK_DW_ERROR(dwRelease(m_sdk));
}

void detectAndTrackImpl::initInputPorts() {
  using dw::core::operator""_sv;
  NODE_INIT_INPUT_PORT("IN_IMG"_sv);
}

void detectAndTrackImpl::initOutputPorts() {
  using dw::core::operator""_sv;
  {
    // dw::framework::parameter_traits<YoloScoreRectArray>::SpecimenT ref{};
    // NODE_INIT_OUTPUT_PORT("BOX_ARR"_sv, ref);

    size_t size = m_YoloScoreRectArraySize;
    NODE_INIT_OUTPUT_PORT("BOX_ARR"_sv, size);
    mYoloScoreRectArray.size = size;
    mYoloScoreRectArray.data = m_yoloData.data();
  }
  {
    // dw::framework::parameter_traits<uint32_t>::SpecimenT ref{};
    NODE_INIT_OUTPUT_PORT("BOX_NUM"_sv);  //, ref);
  }
}

void detectAndTrackImpl::registerPasses() {
  using dw::core::operator""_sv;
  NODE_REGISTER_PASS("PROCESS"_sv,
                     [this]() -> dwStatus { return processPass(); });
}

dwStatus detectAndTrackImpl::processPass() {
  auto& inPort0 = NODE_GET_INPUT_PORT("IN_IMG"_sv);
  auto& outPort0 = NODE_GET_OUTPUT_PORT("BOX_ARR"_sv);
  auto& outPort1 = NODE_GET_OUTPUT_PORT("BOX_NUM"_sv);
  DW_LOGD << "KINELOG - D&T - processPass: " << __LINE__ << Logger::State::endl;

  if (inPort0.isBufferAvailable() && outPort0.isBufferAvailable() &&
      outPort1.isBufferAvailable()) {
    ++m_epochCount;
    dwImageCUDA* yuvImage = nullptr;

    DW_LOGD << "[Epoch " << m_epochCount << "]"
            << " Received IN_IMG " << Logger::State::endl;

    // Read from port
    DW_LOGD << "KINELOG - D&T - processPass: " << __LINE__<< Logger::State::endl;
    dwImageHandle_t nextFrame = *inPort0.getBuffer();
    DW_LOGD << "KINELOG - D&T - processPass: " << __LINE__<< Logger::State::endl;
    if (nextFrame != nullptr) {
      DW_LOGD << "KINELOG - D&T - processPass: " << __LINE__<< Logger::State::endl;
      CHECK_DW_ERROR(dwImageTransformation_copyFullImage(
          m_imageRGBA, nextFrame, ImageTransformationEngine_));
      dwImage_getCUDA(&yuvImage, nextFrame);
      DW_LOGD << "KINELOG - D&T - processPass: " << __LINE__<< Logger::State::endl;

      // Run data conditioner to prepare input for the network
      dwImageCUDA* rgbaImage;
      DW_LOGD << "KINELOG - D&T - processPass: " << __LINE__<< Logger::State::endl;
      CHECK_DW_ERROR(dwImage_getCUDA(&rgbaImage, m_imageRGBA));
      DW_LOGD << "KINELOG - D&T - processPass: " << __LINE__<< Logger::State::endl;
      CHECK_DW_ERROR(dwDataConditioner_prepareDataRaw(
          m_dnnInputDevice, &rgbaImage, 1, &m_detectionRegion,
          cudaAddressModeClamp, m_dataConditioner));
      DW_LOGD << "KINELOG - D&T - processPass: " << __LINE__<< Logger::State::endl;

      // Run DNN on the output of data conditioner
      CHECK_DW_ERROR(
          dwDNN_inferRaw(m_dnnOutputsDevice, &m_dnnInputDevice, 1U, m_dnn));
      DW_LOGD << "KINELOG - D&T - processPass: " << __LINE__<< Logger::State::endl;

      // Copy output back
      CHECK_CUDA_ERROR(cudaMemcpy(
          m_dnnOutputsHost[0].get(), m_dnnOutputsDevice[0],
          sizeof(float32_t) * m_totalSizesOutput[0], cudaMemcpyDeviceToHost));
      DW_LOGD << "KINELOG - D&T - processPass: " << __LINE__<< Logger::State::endl;

      // Interpret output blobs to extract detected boxes
      interpretOutput(m_dnnOutputsHost[m_cvgIdx].get(), &m_detectionRegion);
      DW_LOGD << "KINELOG - D&T - processPass: " << __LINE__<< Logger::State::endl;

      // Track objects
      // -------------------------------------------------
      // -------------------------------------------------
      // TO COMMENT TO SHOW BB FROM DETECTION
      runTracker(yuvImage);
      // -------------------------------------------------
      // -------------------------------------------------
      DW_LOGD << "KINELOG - D&T - processPass: " << __LINE__<< Logger::State::endl;

      // Update output size, if num bb is greater than m_YoloScoreRectArraySize
      // cap to it
      int output_size = std::min((int)m_trackedBoxListFloat.size(), (int)m_YoloScoreRectArraySize);
      mYoloScoreRectArray.size = output_size;
      DW_LOGD << "KINELOG - D&T - processPass: " << __LINE__<< Logger::State::endl;

      for (int i = 0; i < output_size; i++) {
        YoloScoreRect ysr;
        ysr.rectf = m_trackedBoxListFloat[i];
        ysr.score = 0;
        ysr.classIndex = 0;
        m_yoloData.at(i) = ysr;
      }
      DW_LOGD << "KINELOG - D&T - processPass: " << __LINE__<< Logger::State::endl;

    } else {
      DW_LOGD << "KINELOG - D&T - processPass: " << __LINE__<< Logger::State::endl;
      mYoloScoreRectArray.size = 0;
      reset_node();
    }

    DW_LOGD << "KINELOG - D&T - processPass: " << __LINE__<< Logger::State::endl;
    *outPort0.getBuffer() = mYoloScoreRectArray;
    *outPort1.getBuffer() = mYoloScoreRectArray.size;
    outPort0.send();
    outPort1.send();
  }
  return DW_SUCCESS;
}

void detectAndTrackImpl::initialize() {
  // -----------------------------------------
  // Initialize DriveWorks SDK context and SAL
  // -----------------------------------------
  DW_LOGD << "KINELOG - D&T - Initialize DriveWorks SDK context and SAL: "<< __LINE__ << Logger::State::endl;
  {
    // initialize SDK context, using data folder
    dwContextParameters sdkParams = {};
    CHECK_DW_ERROR(dwInitialize(&m_sdk, DW_VERSION, &sdkParams));
  }

  //------------------------------------------------------------------------------
  // initialize Sensors
  //------------------------------------------------------------------------------
  DW_LOGD << "KINELOG - D&T - Initialize Sensors: " << __LINE__<< Logger::State::endl;
  {
    dwImageProperties
        displayProperties;  // = m_camera->getOutputProperties();  // TODO

    displayProperties.height = 2168;
    displayProperties.width = 3848;
    displayProperties.memoryLayout = DW_IMAGE_MEMORY_TYPE_PITCH;
    displayProperties.type = DW_IMAGE_CUDA;
    displayProperties.format = DW_IMAGE_FORMAT_RGBA_UINT8;

    displayProperties.width = displayProperties.width / 1;
    displayProperties.height = displayProperties.height / 1;

    CHECK_DW_ERROR(dwImage_create(&m_imageRGBA, displayProperties, m_sdk));

    m_imageWidth = displayProperties.width;
    m_imageHeight = displayProperties.height;
  }

  //------------------------------------------------------------------------------
  // initialize Image Transformation Engine
  //------------------------------------------------------------------------------
  {
    DW_LOGD << "KINELOG - D&T - Initialize Image Transformation Engine: "<< __LINE__ << Logger::State::endl;
    dwImageTransformation_initialize(&ImageTransformationEngine_, {}, m_sdk);
    dwImageTransformation_setInterpolationMode(
        DW_IMAGEPROCESSING_INTERPOLATION_DEFAULT, ImageTransformationEngine_);
  }

  //------------------------------------------------------------------------------
  // initialize DNN
  //------------------------------------------------------------------------------
  DW_LOGD << "KINELOG - D&T - Initialize DNN: " << __LINE__<< Logger::State::endl;
  {
    // If not specified, load the correct network based on platform
    std::string tensorRTModel =
        "/home/kineton/Vincenzo_CGF/test/yolo.bin";  // TODO shouldn't be hardcoded

    // Initialize DNN from a TensorRT file
    CHECK_DW_ERROR(dwDNN_initializeTensorRTFromFile(
        &m_dnn, tensorRTModel.c_str(), nullptr, DW_PROCESSOR_TYPE_GPU, m_sdk));

    CHECK_DW_ERROR(dwDNN_setCUDAStream(m_cudaStream, m_dnn));

    auto getTotalSize = [](const dwBlobSize& blobSize) {
      return blobSize.channels * blobSize.height * blobSize.width;
    };

    // Get input dimensions
    CHECK_DW_ERROR(dwDNN_getInputSize(&m_networkInputDimensions, 0U, m_dnn));
    // Calculate total size needed to store input
    m_totalSizeInput = getTotalSize(m_networkInputDimensions);
    // Allocate GPU memory
    CHECK_CUDA_ERROR(cudaMalloc((void**)&m_dnnInputDevice,
                                sizeof(float32_t) * m_totalSizeInput));

    // Get output dimensions
    CHECK_DW_ERROR(
        dwDNN_getOutputSize(&m_networkOutputDimensions[0], 0U, m_dnn));

    // Calculate total size needed to store output
    m_totalSizesOutput[0] = getTotalSize(m_networkOutputDimensions[0]);

    // Get coverage and bounding box blob indices
    const char* coverageBlobName = "output0";
    CHECK_DW_ERROR(dwDNN_getOutputIndex(&m_cvgIdx, coverageBlobName, m_dnn));

    // Allocate GPU memory
    CHECK_CUDA_ERROR(cudaMalloc((void**)&m_dnnOutputsDevice[0],
                                sizeof(float32_t) * m_totalSizesOutput[0]));

    // Allocate CPU memory for reading the output of DNN
    m_dnnOutputsHost[0].reset(new float32_t[m_totalSizesOutput[0]]);

    // Get metadata from DNN module
    // DNN loads metadata automatically from json file stored next to the dnn
    // model, with the same name but additional .json extension if present.
    // Otherwise, the metadata will be filled with default values and the
    // dataconditioner parameters should be filled manually.
    dwDNNMetaData metadata;
    CHECK_DW_ERROR(dwDNN_getMetaData(&metadata, m_dnn));

    // Initialie data conditioner
    CHECK_DW_ERROR(dwDataConditioner_initialize(
        &m_dataConditioner, &m_networkInputDimensions, 1U,
        &metadata.dataConditionerParams, m_cudaStream, m_sdk));

    // Detection region
    m_detectionRegion.width = std::min(
        static_cast<uint32_t>(m_networkInputDimensions.width), m_imageWidth);
    m_detectionRegion.height = std::min(
        static_cast<uint32_t>(m_networkInputDimensions.height), m_imageHeight);
    m_detectionRegion.x = (m_imageWidth - m_detectionRegion.width) / 2;
    m_detectionRegion.y = (m_imageHeight - m_detectionRegion.height) / 2;
  }

  //------------------------------------------------------------------------------
  // Initialize Feature Tracker
  //------------------------------------------------------------------------------
  {
    DW_LOGD << "KINELOG - D&T - Initialize Feature Tracker: " << __LINE__<< Logger::State::endl;
    m_maxFeatureCount = 2000;
    m_historyCapacity = 3;

    // Feature Detector
    dwFeature2DDetectorConfig featureDetectorConfig{};
    featureDetectorConfig.imageWidth = m_imageWidth;
    featureDetectorConfig.imageHeight = m_imageHeight;
    CHECK_DW_ERROR(
        dwFeature2DDetector_initDefaultParams(&featureDetectorConfig));
    featureDetectorConfig.maxFeatureCount = m_maxFeatureCount;
    CHECK_DW_ERROR(dwFeature2DDetector_initialize(
        &m_featureDetector, &featureDetectorConfig, m_cudaStream, m_sdk));

    // Feature Tracker
    dwFeature2DTrackerConfig featureTrackerConfig{};
    featureTrackerConfig.imageWidth = m_imageWidth;
    featureTrackerConfig.imageHeight = m_imageHeight;
    CHECK_DW_ERROR(dwFeature2DTracker_initDefaultParams(&featureTrackerConfig));
    featureTrackerConfig.maxFeatureCount = m_maxFeatureCount;
    featureTrackerConfig.historyCapacity = m_historyCapacity;
    featureTrackerConfig.detectorType = featureDetectorConfig.type;
    CHECK_DW_ERROR(dwFeature2DTracker_initialize(
        &m_featureTracker, &featureTrackerConfig, m_cudaStream, m_sdk));

    // Tracker pyramid init
    CHECK_DW_ERROR(dwPyramid_create(
        &m_pyramidPrevious, featureTrackerConfig.pyramidLevelCount,
        m_imageWidth, m_imageHeight, DW_TYPE_UINT8, m_sdk));
    CHECK_DW_ERROR(dwPyramid_create(
        &m_pyramidCurrent, featureTrackerConfig.pyramidLevelCount, m_imageWidth,
        m_imageHeight, DW_TYPE_UINT8, m_sdk));

    CHECK_DW_ERROR(dwFeatureHistoryArray_createNew(&m_featureHistoryCPU, m_maxFeatureCount, m_historyCapacity, DW_MEMORY_TYPE_CPU, nullptr, m_sdk));
    CHECK_DW_ERROR(dwFeatureHistoryArray_createNew(&m_featureHistoryGPU, m_maxFeatureCount, m_historyCapacity, DW_MEMORY_TYPE_CUDA, nullptr, m_sdk));
    CHECK_DW_ERROR(dwFeatureArray_createNew(&m_featureDetectedGPU, m_maxFeatureCount, DW_MEMORY_TYPE_CUDA, nullptr, m_sdk));

    // Set up mask. Apply feature tracking only to half of the image
    dwPyramidImageProperties pyramidProps{};
    CHECK_DW_ERROR(
        dwPyramid_getProperties(&pyramidProps, &m_pyramidCurrent, m_sdk));
    m_maskSize.x =
        pyramidProps.levelProps[featureDetectorConfig.detectionLevel].width;
    m_maskSize.y =
        pyramidProps.levelProps[featureDetectorConfig.detectionLevel].height;
    CHECK_CUDA_ERROR(cudaMallocPitch(&m_featureMask, &m_maskPitch, m_maskSize.x,
                                     m_maskSize.y));
    CHECK_CUDA_ERROR(
        cudaMemset(m_featureMask, 255, m_maskPitch * m_maskSize.y));
    CHECK_CUDA_ERROR(
        cudaMemset(m_featureMask, 0, m_maskPitch * m_maskSize.y / 2));

    CHECK_DW_ERROR(dwFeature2DDetector_setMask(m_featureMask, m_maskPitch,
                                               m_maskSize.x, m_maskSize.y,
                                               m_featureDetector));
  }

  //------------------------------------------------------------------------------
  // Initialize Box Tracker
  //------------------------------------------------------------------------------
  {
    DW_LOGD << "KINELOG - D&T - Initialize Box Tracker: " << __LINE__<< Logger::State::endl;
    dwBoxTracker2DParams params{};
    dwBoxTracker2D_initParams(&params);
    params.maxBoxImageScale = 0.5f;
    params.minBoxImageScale = 0.005f;
    params.similarityThreshold = 0.2f;
    params.groupThreshold = 2.0f;
    params.maxBoxCount = m_maxDetections;
    CHECK_DW_ERROR(dwBoxTracker2D_initialize(
        &m_boxTracker, &params, m_imageWidth, m_imageHeight, m_sdk));
    // Reserve for storing feature locations and statuses in CPU
    m_currentFeatureLocations.reserve(2 * m_maxFeatureCount);
    m_previousFeatureLocations.reserve(2 * m_maxFeatureCount);
    m_featureStatuses.reserve(2 * m_maxFeatureCount);
    m_trackedBoxListFloat.reserve(m_maxDetections);
  }
}

void detectAndTrackImpl::reset_node() {
  CHECK_DW_ERROR(dwDNN_reset(m_dnn));
  CHECK_DW_ERROR(dwDataConditioner_reset(m_dataConditioner));

  CHECK_DW_ERROR(
      dwFeatureHistoryArray_reset(&m_featureHistoryGPU, m_cudaStream));
  CHECK_DW_ERROR(dwFeatureArray_reset(&m_featureDetectedGPU, m_cudaStream));

  CHECK_DW_ERROR(dwFeature2DDetector_reset(m_featureDetector));
  CHECK_DW_ERROR(dwFeature2DTracker_reset(m_featureTracker));
  CHECK_DW_ERROR(dwBoxTracker2D_reset(m_boxTracker));

  CHECK_DW_ERROR(dwFeature2DDetector_setMask(m_featureMask, m_maskPitch,
                                             m_maskSize.x, m_maskSize.y,
                                             m_featureDetector));
}


/**
 * @brief calculate the IOU(Intersection over Union) of two boxes.
 *
 * @param[in] box1 The decription of box one.
 * @param[in] box2 The decription of box two.
 * @retval IOU value.
 */
float32_t detectAndTrackImpl::calculateIouOfBoxes(dwRectf box1, dwRectf box2) {
  float32_t x1 = std::max(box1.x, box2.x);
  float32_t y1 = std::max(box1.y, box2.y);
  float32_t x2 = std::min(box1.x + box1.width, box2.x + box2.width);
  float32_t y2 = std::min(box1.y + box1.height, box2.y + box2.height);
  float32_t w = std::max(0.0f, x2 - x1);
  float32_t h = std::max(0.0f, y2 - y1);
  float32_t over_area = w * h;
  return float32_t(over_area) / float32_t(box1.width * box1.height +
                                          box2.width * box2.height - over_area);
}

/**
 * @brief do nms(non maximum suppression) for Yolo output boxes.
 *
 * @param[in] boxes The boxes which are going to be operated with nms.
 * @param[in] threshold The threshold. Used in nms to delete duplicate boxes.
 * @return The boxes which have been operated with nms.
 */
YoloScoreRectVector detectAndTrackImpl::nonMaximumSuppression(
    YoloScoreRectVector& boxes, float32_t threshold) {
  YoloScoreRectVector results;
  std::sort(boxes.begin(), boxes.end(), sort_score);
  while (boxes.size() > 0) {
    results.push_back(boxes[0]);
    uint32_t index = 1;
    while (index < boxes.size()) {
      float32_t iou_value =
          calculateIouOfBoxes(boxes[0].rectf, boxes[index].rectf);
      if (iou_value > threshold) {
        boxes.erase(boxes.begin() + index);
      } else {
        index++;
      }
    }
    boxes.erase(boxes.begin());
  }
  return results;
}

//------------------------------------------------------------------------------
void detectAndTrackImpl::interpretOutput(const float32_t* outConf,
                                         const dwRect* const roi) {
  // Clear detection list
  m_detectedTrackBox.clear();
  m_label.clear();

  uint32_t numBBoxes = 0U;
  uint16_t gridH = m_networkOutputDimensions[0].height;
  uint16_t gridW = m_networkOutputDimensions[0].width;
  YoloScoreRectVector tmpRes;

  for (uint16_t gridY = 0U; gridY < gridH; ++gridY) {
    const float32_t* outConfRow = &outConf[gridY * gridW];
    if (outConfRow[4] < CONFIDENCE_THRESHOLD || numBBoxes >= 100) {
      continue;
    }
    uint16_t maxIndex = 0;
    float32_t maxScore = 0;
    for (uint16_t i = 5; i < 85; i++) {
      // The col 5-85 represents the probability of each class.
      if (outConfRow[i] > maxScore) {
        maxScore = outConfRow[i];
        maxIndex = i;
      }
    }

    if (maxScore > SCORE_THRESHOLD) {
      // This is a detection!
      float32_t imageX = (float32_t)outConfRow[0];
      float32_t imageY = (float32_t)outConfRow[1];
      float32_t bboxW = (float32_t)outConfRow[2];
      float32_t bboxH = (float32_t)outConfRow[3];

      float32_t boxX1Tmp = (float32_t)(imageX - 0.5 * bboxW);
      float32_t boxY1Tmp = (float32_t)(imageY - 0.5 * bboxH);
      float32_t boxX2Tmp = (float32_t)(imageX + 0.5 * bboxW);
      float32_t boxY2Tmp = (float32_t)(imageY + 0.5 * bboxH);

      float32_t boxX1;
      float32_t boxY1;
      float32_t boxX2;
      float32_t boxY2;

      dwDataConditioner_outputPositionToInput(&boxX1, &boxY1, boxX1Tmp,
                                              boxY1Tmp, roi, m_dataConditioner);
      dwDataConditioner_outputPositionToInput(&boxX2, &boxY2, boxX2Tmp,
                                              boxY2Tmp, roi, m_dataConditioner);
      dwRectf bboxFloat{boxX1, boxY1, boxX2 - boxX1, boxY2 - boxY1};
      tmpRes.push_back({bboxFloat, maxScore, (uint16_t)(maxIndex - 5)});
      numBBoxes++;
    }
  }

  YoloScoreRectVector tmpResAfterNMS =
      nonMaximumSuppression(tmpRes, float32_t(0.45));

  // -------------------------------------------------
  // -------------------------------------------------
  // TO UNCOMMENT TO SHOW BB FROM DETECTION
  // m_trackedBoxListFloat.clear();
  // -------------------------------------------------
  // -------------------------------------------------
  for (uint32_t i = 0; i < tmpResAfterNMS.size(); i++) {
    YoloScoreRect box = tmpResAfterNMS[i];
    dwRectf bboxFloat = box.rectf;
    dwBox2D bbox;
    bbox.width = static_cast<int32_t>(std::round(bboxFloat.width));
    bbox.height = static_cast<int32_t>(std::round(bboxFloat.height));
    bbox.x = static_cast<int32_t>(std::round(bboxFloat.x));
    bbox.y = static_cast<int32_t>(std::round(bboxFloat.y));

    if (YOLO_CLASS_NAMES[box.classIndex] == "car" ||
        YOLO_CLASS_NAMES[box.classIndex] == "person" ||
        YOLO_CLASS_NAMES[box.classIndex] == "motorcycle" ||
        YOLO_CLASS_NAMES[box.classIndex] == "traffic light" ||
        YOLO_CLASS_NAMES[box.classIndex] == "bus" ||
        YOLO_CLASS_NAMES[box.classIndex] == "bicycle" ||
        YOLO_CLASS_NAMES[box.classIndex] == "stop sign" ||
        YOLO_CLASS_NAMES[box.classIndex] == "truck") {
      m_label.push_back(YOLO_CLASS_NAMES[box.classIndex]);
      dwTrackedBox2D tmpTrackBox;
      tmpTrackBox.box = bbox;
      tmpTrackBox.id = -1;
      tmpTrackBox.confidence = box.score;
      m_detectedTrackBox.push_back(tmpTrackBox);

      // -------------------------------------------------
      // -------------------------------------------------
      // -------------------------------------------------
      // TO UNCOMMENT TO SHOW BB FROM DETECTION
      // dwRectf rectf;
      // rectf.x = static_cast<float32_t>(bbox.x);
      // rectf.y = static_cast<float32_t>(bbox.y);
      // rectf.width = static_cast<float32_t>(bbox.width);
      // rectf.height = static_cast<float32_t>(bbox.height);
      // m_trackedBoxListFloat.push_back(rectf);
      // -------------------------------------------------
      // -------------------------------------------------
      // -------------------------------------------------
    }
  }
}

//------------------------------------------------------------------------------
float32_t detectAndTrackImpl::overlap(const dwRectf& boxA,
                                      const dwRectf& boxB) {
  int32_t overlapWidth = std::min(boxA.x + boxA.width, boxB.x + boxB.width) -
                         std::max(boxA.x, boxB.x);
  int32_t overlapHeight = std::min(boxA.y + boxA.height, boxB.y + boxB.height) -
                          std::max(boxA.y, boxB.y);

  return (overlapWidth < 0 || overlapHeight < 0)
             ? 0.0f
             : (overlapWidth * overlapHeight);
}

//------------------------------------------------------------------------------
void detectAndTrackImpl::runTracker(const dwImageCUDA* image) {
  // add candidates to box tracker
  DW_LOGD << "KINELOG - D&T - runTracker: " << __LINE__ << Logger::State::endl;
  CHECK_DW_ERROR(dwBoxTracker2D_addPreClustered(
      m_detectedTrackBox.data(), m_detectedTrackBox.size(), m_boxTracker));
  DW_LOGD << "KINELOG - D&T - runTracker: " << __LINE__ << Logger::State::endl;

  // track features
  uint32_t featureCount = trackFeatures(image);
  DW_LOGD << "KINELOG - D&T - runTracker: " << __LINE__ << Logger::State::endl;

  // If this is not the first frame, update the features
  if (m_epochCount != 0) {
    DW_LOGD << "KINELOG - D&T - runTracker: " << __LINE__ << Logger::State::endl;
    // update box features
    CHECK_DW_ERROR(dwBoxTracker2D_updateFeatures(
        m_previousFeatureLocations.data(), m_featureStatuses.data(),
        featureCount, m_boxTracker));
    DW_LOGD << "KINELOG - D&T - runTracker: " << __LINE__ << Logger::State::endl;
  }

  // Run box tracker
  DW_LOGD << "KINELOG - D&T - runTracker: " << __LINE__ << Logger::State::endl;
  CHECK_DW_ERROR(dwBoxTracker2D_track(
      m_currentFeatureLocations.data(), m_featureStatuses.data(),
      m_previousFeatureLocations.data(), m_boxTracker));
  DW_LOGD << "KINELOG - D&T - runTracker: " << __LINE__ << Logger::State::endl;

  // Get tracked boxes
  CHECK_DW_ERROR(
      dwBoxTracker2D_get(&m_trackedBoxes, &m_numTrackedBoxes, m_boxTracker));
  DW_LOGD << "KINELOG - D&T - runTracker: " << __LINE__ << Logger::State::endl;

  // Extract boxes from tracked object list
  m_trackedBoxListFloat.clear();
  DW_LOGD << "KINELOG - D&T - runTracker: " << __LINE__ << Logger::State::endl;
  for (uint32_t tIdx = 0U; tIdx < m_numTrackedBoxes; ++tIdx) {
    DW_LOGD << "KINELOG - D&T - runTracker: " << __LINE__ << Logger::State::endl;
    const dwBox2D& box = m_trackedBoxes[tIdx].box;
    dwRectf rectf;
    rectf.x = static_cast<float32_t>(box.x);
    rectf.y = static_cast<float32_t>(box.y);
    rectf.width = static_cast<float32_t>(box.width);
    rectf.height = static_cast<float32_t>(box.height);
    m_trackedBoxListFloat.push_back(rectf);
    DW_LOGD << "KINELOG - D&T - runTracker: " << __LINE__ << Logger::State::endl;
  }
  DW_LOGD << "KINELOG - D&T - runTracker: " << __LINE__ << Logger::State::endl;
}

// ------------------------------------------------
// Feature tracking
// ------------------------------------------------
uint32_t detectAndTrackImpl::trackFeatures(const dwImageCUDA* image) {
  DW_LOGD << "KINELOG - D&T - trackFeatures: " << __LINE__<< Logger::State::endl;
  std::swap(m_pyramidCurrent, m_pyramidPrevious);
  DW_LOGD << "KINELOG - D&T - trackFeatures: " << __LINE__<< Logger::State::endl;

  // build pyramid
  CHECK_DW_ERROR(dwImageFilter_computePyramid(&m_pyramidCurrent, image,
                                              m_cudaStream, m_sdk));
  DW_LOGD << "KINELOG - D&T - trackFeatures: " << __LINE__<< Logger::State::endl;

  // track features
  dwFeatureArray featurePredicted{};
  DW_LOGD << "KINELOG - D&T - trackFeatures: " << __LINE__<< Logger::State::endl;
  CHECK_DW_ERROR(dwFeature2DTracker_trackFeatures(
      &m_featureHistoryGPU, &featurePredicted, nullptr, &m_featureDetectedGPU,
      nullptr, &m_pyramidPrevious, &m_pyramidCurrent, m_featureTracker));
  DW_LOGD << "KINELOG - D&T - trackFeatures: " << __LINE__<< Logger::State::endl;

  // Get feature info to CPU
  DW_LOGD << "before dwFeatureHistoryArray_copyAsync" << Logger::State::endl;
  // ####### LINE 658 is causing the problem #######
  CHECK_DW_ERROR(dwFeatureHistoryArray_copyAsync(&m_featureHistoryCPU, &m_featureHistoryGPU, 0));
  DW_LOGD << "after dwFeatureHistoryArray_copyAsync" << Logger::State::endl;

  // Update feature locations after tracking
  uint32_t featureCount = updateFeatureLocationsStatuses();
  DW_LOGD << "KINELOG - D&T - trackFeatures: " << __LINE__<< Logger::State::endl;

  // detect new features
  CHECK_DW_ERROR(dwFeature2DDetector_detectFromPyramid(
      &m_featureDetectedGPU, &m_pyramidCurrent, &featurePredicted, nullptr,
      m_featureDetector));
  DW_LOGD << "KINELOG - D&T - trackFeatures: " << __LINE__<< Logger::State::endl;

  return featureCount;
  // return 0;
}

// ------------------------------------------------
uint32_t detectAndTrackImpl::updateFeatureLocationsStatuses() {
  // Get previous locations and update box tracker
  dwFeatureArray curFeatures{};
  dwFeatureArray preFeatures{};
  CHECK_DW_ERROR(dwFeatureHistoryArray_getCurrent(&curFeatures, &m_featureHistoryCPU));
  CHECK_DW_ERROR(dwFeatureHistoryArray_getPrevious(&preFeatures, &m_featureHistoryCPU));

  dwVector2f* preLocations = preFeatures.locations;
  dwVector2f* curLocations = curFeatures.locations;
  uint32_t newSize = std::min(m_maxFeatureCount, *m_featureHistoryCPU.featureCount);

  m_previousFeatureLocations.clear();
  m_currentFeatureLocations.clear();
  m_featureStatuses.clear();
  for (uint32_t featureIdx = 0; featureIdx < newSize; featureIdx++) {
    m_previousFeatureLocations.push_back(preLocations[featureIdx].x);
    m_previousFeatureLocations.push_back(preLocations[featureIdx].y);
    m_currentFeatureLocations.push_back(curLocations[featureIdx].x);
    m_currentFeatureLocations.push_back(curLocations[featureIdx].y);

    m_featureStatuses.push_back(m_featureHistoryCPU.statuses[featureIdx]);
  }
  return newSize;
}
}  // namespace framework
}  // namespace dw