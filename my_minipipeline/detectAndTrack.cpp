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

#include "./detectAndTrack.hpp"

#include <dwcgf/parameter/ParameterProvider.hpp>

#include "./detectAndTrackImpl.hpp"

namespace dw {
namespace framework {

std::unique_ptr<detectAndTrack> detectAndTrack::create(
    dw::framework::ParameterProvider& provider) {
  // for custom logic use the following:
  // auto constructorArguments =
  // dw::framework::createConstructorArguments<detectAndTrack>();
  // YourFirstArgumentType& params = std::get<0>(constructorArguments);
  // params.xxx = provider.getRequired(...);
  // dw::framework::populateParameters<detectAndTrack>(constructorArguments,
  // provider); return
  // dw::framework::makeUniqueFromTuple<detectAndTrack>(std::move(constructorArguments));

  return dw::framework::create<detectAndTrack>(provider);
}

detectAndTrack::detectAndTrack(const dwContextHandle_t ctx)
    : dw::framework::ExceptionSafeProcessNode(
          std::make_unique<detectAndTrackImpl>(ctx)) {}

}  // namespace framework
}  // namespace dw

#include <dwcgf/node/NodeFactory.hpp>

DW_REGISTER_NODE(dw::framework::detectAndTrack)
