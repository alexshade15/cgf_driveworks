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
// consequences of use of such information or for any infringement of patents or
// other rights of third parties that may result from its use. No license is
// granted by implication or otherwise under any patent or patent rights of
// NVIDIA CORPORATION & AFFILIATES. No third party distribution is allowed
// unless expressly authorized by NVIDIA. Details are subject to change without
// notice. This code supersedes and replaces all information previously
// supplied. NVIDIA CORPORATION & AFFILIATES products are not authorized for use
// as critical components in life support devices or systems without express
// written approval of NVIDIA CORPORATION & AFFILIATES.
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

#ifndef DW_FRAMEWORK_DETECTANDTRACK_HPP_
#define DW_FRAMEWORK_DETECTANDTRACK_HPP_

#include <dw/image/Image.h>

#include <dwcgf/node/Node.hpp>
#include <dwcgf/node/impl/ExceptionSafeNode.hpp>
#include <dwcgf/parameter/ParameterDescriptor.hpp>
#include <dwcgf/pass/Pass.hpp>
#include <dwcgf/pass/PassDescriptor.hpp>
#include <dwcgf/port/Port.hpp>
#include <dwcgf/port/PortDescriptor.hpp>
#include <dwframework/dwnodes/common/channelpackets/Image.hpp>
#include <framework/DriveWorksSample.hpp>

#include "YoloNet.hpp"

namespace dw {
namespace framework {

bool sort_score(YoloScoreRect, YoloScoreRect);

typedef std::vector<YoloScoreRect> YoloScoreRectVector;

class detectAndTrackImpl;

class detectAndTrack : public dw::framework::ExceptionSafeProcessNode {
 public:
  static constexpr char LOG_TAG[] = "detectAndTrack";

  static constexpr auto describeInputPorts() {
    using dw::core::operator""_sv;
    return dw::framework::describePortCollection(
        DW_DESCRIBE_PORT(dwImageHandle_t, "IN_IMG"_sv));
  }

  static constexpr auto describeOutputPorts() {
    using dw::core::operator""_sv;
    return dw::framework::describePortCollection(
        DW_DESCRIBE_PORT(YoloScoreRectArray, "BOX_ARR"_sv),
        DW_DESCRIBE_PORT(uint32_t, "BOX_NUM"_sv));
  }

  static constexpr auto describePasses() {
    using dw::core::operator""_sv;
    return dw::framework::describePassCollection(
        dw::framework::describePass("SETUP"_sv, DW_PROCESSOR_TYPE_CPU),
        dw::framework::describePass("PROCESS"_sv, DW_PROCESSOR_TYPE_GPU),
        dw::framework::describePass("TEARDOWN"_sv, DW_PROCESSOR_TYPE_CPU));
  }

  static constexpr auto describeParameters() {
    using dw::core::operator""_sv;
    return dw::framework::describeConstructorArguments<dwContextHandle_t>(
        dw::framework::describeConstructorArgument(
            DW_DESCRIBE_UNNAMED_PARAMETER(dwContextHandle_t)));
  }

  static std::unique_ptr<detectAndTrack> create(
      dw::framework::ParameterProvider& provider);

  detectAndTrack(const dwContextHandle_t ctx);
};

}  // namespace framework
}  // namespace dw

#endif  // DW_FRAMEWORK_DETECTANDTRACK_HPP_
