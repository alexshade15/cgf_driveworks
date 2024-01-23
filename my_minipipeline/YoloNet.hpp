#ifndef AUTOWARE_AUTO_MAPPING_MSGS_CHANNEL_YOLO_HPP
#define AUTOWARE_AUTO_MAPPING_MSGS_CHANNEL_YOLO_HPP

#include <dw/core/base/Types.h>

#include <dwcgf/channel/ChannelPacketTypes.hpp>
#include <framework/DriveWorksSample.hpp>

#define DWFRAMEWORK_DECLARE_PACKET_DWTYPE_RELATION(DATA_TYPE, SPECIMEN_TYPE, \
                                                   ENUM_SPEC)                \
  DWFRAMEWORK_DECLARE_PACKET_TYPE_RELATION(                                  \
      DATA_TYPE, SPECIMEN_TYPE,                                              \
      dw::framework::DWChannelPacketTypeID::ENUM_SPEC)

#define DWFRAMEWORK_DECLARE_PACKET_DWTYPE_RELATION_SIMPLE(DATA_TYPE, \
                                                          ENUM_SPEC) \
  DWFRAMEWORK_DECLARE_PACKET_DWTYPE_RELATION(DATA_TYPE, DATA_TYPE, ENUM_SPEC)

namespace dw {
namespace framework {

enum dwSerializationTypeExt {
  DW_HAD_MAP_BIN = 257,
  DW_SCENARIO = 258,
  DW_LANELET_ROUTE = 259,
  DW_YOLO = 260,
  DW_CUSTOM_RAW_BUFFER = 1024
};

}  // namespace framework
}  // namespace dw

static constexpr size_t YOLO_MAX_BOX_NUM = 64;

typedef struct YoloScoreRect {
  dwRectf rectf;
  float32_t score;
  uint16_t classIndex;
} YoloScoreRect;
DWFRAMEWORK_DECLARE_PACKET_TYPE_POD(YoloScoreRect)

typedef struct YoloScoreRectArray {
  size_t size;
  YoloScoreRect* data;
} YoloScoreRectArray;

constexpr dw::framework::ChannelPacketTypeID YoloTypeID =
    dw::framework::dwSerializationTypeExt::DW_YOLO;

DWFRAMEWORK_DECLARE_PACKET_TYPE_RELATION(YoloScoreRectArray, size_t,
                                         YoloTypeID);

#endif  // YOLO_HPP
