/*
 * Copyright (c) 2025 Xu Yang
 * HKUST UAV Group
 *
 * Author: Xu Yang
 * Affiliation: HKUST UAV Group
 * Date: April 2025
 * License: MIT License
 */

#ifndef TOFSENSEMIMPL_H
#define TOFSENSEMIMPL_H

#include "protocol_extracter/nprotocol_extracter.h"
#include "nlink_unpack/nlink_tofsensem_frame0.h"
#include <serial/serial.h>
#include <unordered_map>
#include <functional>
#include <thread>
#include <atomic>

namespace tofsensem {
class Impl {
public:
  explicit Impl(std::function<void(ntsm_frame0_t)> callback_fun, 
                const std::string &port_name = "/dev/ttyUSB0",
                int baud_rate = 921600);
  ~Impl();

private:
  std::thread m_thread_;
  std::atomic<bool> m_running_;
  NProtocolExtracter extracter_;
  serial::Serial serial_;

  void ImplFrame0(std::function<void(ntsm_frame0_t)> callback_fun);
};

} // namespace tofsensem
#endif // TOFSENSEMIMPL_H
