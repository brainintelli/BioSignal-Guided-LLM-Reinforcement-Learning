#pragma once
#include <cstdint>
#include <cstdlib>
#include <cstring>
namespace maxlab {
enum class Status { MAXLAB_OK=0, MAXLAB_API_FAIL=1, MAXLAB_NO_FRAME=9 };
enum class FilterType { FIR=0 };
struct SpikeEvent { unsigned long frameNo; float amp; uint16_t channel; unsigned char wellId; };
struct FilteredFrameData { uint64_t spikeCount; const SpikeEvent* spikeEvents; };
inline bool mode(const char* value) { return std::getenv("MEA_STUB") && std::strcmp(std::getenv("MEA_STUB"), value)==0; }
inline Status DataStreamerFiltered_open(FilterType) { return mode("open_fail") ? Status::MAXLAB_API_FAIL : Status::MAXLAB_OK; }
inline Status DataStreamerFiltered_close() { return mode("close_fail") ? Status::MAXLAB_API_FAIL : Status::MAXLAB_OK; }
inline Status DataStreamerFiltered_receiveNextFrame(FilteredFrameData* data) {
    static int calls=0; static SpikeEvent event{123456, -3.125f, 42, 0};
    static int empty_calls=0;
    if (mode("empty_frames") && empty_calls++ < 20000) {
        data->spikeCount=0; data->spikeEvents=nullptr; return Status::MAXLAB_OK;
    }
    if (mode("receive_fail")) return Status::MAXLAB_API_FAIL;
    if (calls++==0) { data->spikeCount=1; data->spikeEvents=&event; return Status::MAXLAB_OK; }
    return Status::MAXLAB_NO_FRAME;
}
}
