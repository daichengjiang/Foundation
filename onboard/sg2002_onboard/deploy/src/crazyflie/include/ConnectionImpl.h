/*
 * Copyright (c) 2025 Xu Yang
 * HKUST UAV Group
 *
 * Author: Xu Yang
 * Affiliation: HKUST UAV Group
 * Date: April 2025
 * License: MIT License
 */

#pragma once

#include <string>
#include <queue>
#include <mutex>
#include <condition_variable>
#include "Packet.h"

namespace crazyflie {

struct ConnectionImpl {
    struct Statistics {
        size_t enqueued_count = 0;
        size_t sent_count = 0;
        size_t receive_count = 0;
    };

    std::string uri_;
    std::string runtime_error_;
    
    std::mutex queue_send_mutex_;
    std::priority_queue<std::vector<uint8_t>> queue_send_;
    
    std::mutex queue_recv_mutex_;
    std::condition_variable queue_recv_cv_;
    std::priority_queue<Packet> queue_recv_;
    
    Statistics statistics_;
};

} // namespace crazyflie