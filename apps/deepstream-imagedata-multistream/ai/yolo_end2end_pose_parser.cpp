#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

#include "nvdsinfer_custom_impl.h"

namespace {
const NvDsInferLayerInfo* output0(
    const std::vector<NvDsInferLayerInfo>& layers) {
  for (const auto& layer : layers) {
    if (layer.layerName != nullptr &&
        std::strcmp(layer.layerName, "output0") == 0) {
      return &layer;
    }
  }
  return nullptr;
}
}  // namespace

extern "C" bool NvDsInferParseYoloEnd2EndPose(
    const std::vector<NvDsInferLayerInfo>& layers,
    const NvDsInferNetworkInfo& network,
    const NvDsInferParseDetectionParams& params,
    std::vector<NvDsInferObjectDetectionInfo>& objects) {
  const auto* layer = output0(layers);
  if (layer == nullptr || layer->buffer == nullptr) {
    std::cerr << "pose parser: output0 unavailable" << std::endl;
    return false;
  }

  const auto& dims = layer->inferDims;
  if (dims.numDims != 2 || dims.d[0] != 300 || dims.d[1] != 57) {
    std::cerr << "pose parser: expected output0 [300,57], got ";
    for (unsigned int i = 0; i < dims.numDims; ++i) {
      std::cerr << (i ? "x" : "") << dims.d[i];
    }
    std::cerr << std::endl;
    return false;
  }

  const auto* values = static_cast<const float*>(layer->buffer);
  for (unsigned int row = 0; row < 300; ++row) {
    const float* detection = values + row * 57;
    const float confidence = detection[4];
    const int class_id = static_cast<int>(std::lround(detection[5]));
    if (!std::isfinite(confidence) || class_id != 0) {
      continue;
    }
    const float threshold = params.perClassPreclusterThreshold.empty()
        ? 0.25F : params.perClassPreclusterThreshold[0];
    if (confidence < threshold) {
      continue;
    }

    const float left = std::clamp(detection[0], 0.0F,
                                  static_cast<float>(network.width));
    const float top = std::clamp(detection[1], 0.0F,
                                 static_cast<float>(network.height));
    const float right = std::clamp(detection[2], 0.0F,
                                   static_cast<float>(network.width));
    const float bottom = std::clamp(detection[3], 0.0F,
                                    static_cast<float>(network.height));
    if (right <= left || bottom <= top) {
      continue;
    }

    NvDsInferObjectDetectionInfo object{};
    object.classId = 0;
    object.left = left;
    object.top = top;
    object.width = right - left;
    object.height = bottom - top;
    object.detectionConfidence = confidence;
    objects.push_back(object);
  }
  return true;
}

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseYoloEnd2EndPose);
