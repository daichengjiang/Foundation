// Copyright (c) 2025 Xu Yang
// HKUST UAV Group
//
// Author: Xu Yang
// Affiliation: HKUST UAV Group
// Date: April 2025
// License: MIT License

#include "nlink_protocol.h"

#include <assert.h>

#include <numeric>
#include <string>

void NLinkProtocol::HandleData(const uint8_t *data) {
  UnpackFrameData(data);
  assert(HandleDataCallback_);
  HandleDataCallback_();
}

bool NLinkProtocol::Verify(const uint8_t *data) {
  uint8_t sum = 0;
  return data[length() - 1] ==
         std::accumulate(data, data + length() - sizeof(sum), sum);
}

bool NLinkProtocolVLength::UpdateLength(const uint8_t *data,
                                        size_t available_bytes) {
  if (available_bytes < 4)
    return false;
  return set_length(static_cast<size_t>(data[2] | data[3] << 8));
}
