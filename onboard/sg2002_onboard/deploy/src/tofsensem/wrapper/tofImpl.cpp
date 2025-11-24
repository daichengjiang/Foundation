// Copyright (c) 2025 Xu Yang
// HKUST UAV Group
//
// Author: Xu Yang
// Affiliation: HKUST UAV Group
// Date: April 2025
// License: MIT License

#include "tofImpl.h"
#include "nlink_protocol.h"
#include "nlink_unpack/nlink_utils.h"
#include "init_serial.h"

#include <iostream>
#include <vector>

namespace tofsensem {

struct Pixel {
  float dis;
  uint8_t dis_status;
  float signal_strength;
};

struct TofsenseMFrame0 {
  uint32_t id;
  uint64_t system_time;
  std::vector<Pixel> pixels;
};

TofsenseMFrame0 g_msg_tofmframe0;

namespace {

class ProtocolFrame0 : public NLinkProtocolVLength {
public:
  ProtocolFrame0()
      : NLinkProtocolVLength(
            true, g_ntsm_frame0.fixed_part_size,
            {g_ntsm_frame0.frame_header, g_ntsm_frame0.function_mark}) {}

protected:
  bool UpdateLength(const uint8_t *data, size_t available_bytes) override {
    if (available_bytes < g_ntsm_frame0.fixed_part_size)
      return false;
    return set_length(tofm_frame0_size(data));
  }
  void UnpackFrameData(const uint8_t *data) override {
    g_ntsm_frame0.UnpackData(data, length());
  }
};

} // namespace

Impl::Impl(std::function<void(ntsm_frame0_t)> callback_fun, 
            const std::string &port_name, int baud_rate) {
  initSerial(&serial_, port_name, baud_rate);
  ImplFrame0(callback_fun);

  m_running_ = true;
  m_thread_ = std::thread([this]() {
    while (m_running_) {
      auto available_bytes = serial_.available();
      if (available_bytes) {
        std::string str_received;
        serial_.read(str_received, available_bytes);
        extracter_.AddNewData(str_received);
      } else {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  });
}

Impl::~Impl() {
  m_running_ = false;
  if (m_thread_.joinable()) {
    m_thread_.join();
  }
}

void Impl::ImplFrame0(std::function<void(ntsm_frame0_t)> callback_fun) {
  static auto protocol = new ProtocolFrame0;
  extracter_.AddProtocol(protocol);

  protocol->SetHandleDataCallback([=] {
    callback_fun(g_ntsm_frame0);

    // const auto &data = g_ntsm_frame0;
    // g_msg_tofmframe0.id = data.id;
    // g_msg_tofmframe0.system_time = data.system_time;
    // g_msg_tofmframe0.pixels.resize(data.pixel_count);

    // for (int i = 0; i < data.pixel_count; ++i) {
    //   const auto &src_pixel = data.pixels[i];
    //   auto &pixel = g_msg_tofmframe0.pixels[i];
    //   pixel.dis = src_pixel.dis;
    //   pixel.dis_status = src_pixel.dis_status;
    //   pixel.signal_strength = src_pixel.signal_strength;
    // }

    // std::cout << std::endl << "[TofsenseMFrame0] id: " << g_msg_tofmframe0.id
    //           << ", system_time: " << g_msg_tofmframe0.system_time << "\n";
    // std::cout << "dis:\n";
    // for (int i = 0; i < 8; i++) {
    //   for (int j = 0; j < 8; j++) {
    //     int idx = i * 8 + j;
    //     std::cout << (int)(g_msg_tofmframe0.pixels[idx].dis) / 1000.0 << (j < 7 ? " " : "");
    //   }
    //   std::cout << "\n";
    // }

    // std::cout << "signal_strength:\n";
    // for (int i = 0; i < 8; i++) {
    //   for (int j = 0; j < 8; j++) {
    //     int idx = i * 8 + j;
    //     std::cout << g_msg_tofmframe0.pixels[idx].signal_strength << (j < 7 ? " " : "");
    //   }
    //   std::cout << "\n";
    // }
  });
}

} // namespace tofsensem
