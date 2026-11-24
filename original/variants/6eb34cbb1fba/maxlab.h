#pragma once
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <chrono>
namespace maxlab {
enum class Status { MAXLAB_OK=0, MAXLAB_API_FAIL=1, MAXLAB_NO_FRAME=9 };
enum class FilterType { FIR=0 };
struct SpikeEvent { unsigned long frameNo; float amp; uint16_t channel; unsigned char wellId; };
struct FrameInfo { uint64_t frame_number; uint8_t well_id; bool corrupted; };
struct FilteredFrameData { uint64_t spikeCount; const SpikeEvent* spikeEvents; FrameInfo frameInfo{}; };
struct RawFrameData { FrameInfo frameInfo; static const uint64_t amplitudeCount=1024; const float* amplitudes; };
inline bool mode(const char* value) { return std::getenv("MEA_STUB") && std::strcmp(std::getenv("MEA_STUB"), value)==0; }
inline Status DataStreamerFiltered_open(FilterType) { return mode("open_fail") ? Status::MAXLAB_API_FAIL : Status::MAXLAB_OK; }
inline Status DataStreamerFiltered_close() { return mode("close_fail") ? Status::MAXLAB_API_FAIL : Status::MAXLAB_OK; }
inline Status DataStreamerRaw_open() { return mode("raw_open_fail") ? Status::MAXLAB_API_FAIL : Status::MAXLAB_OK; }
inline Status DataStreamerRaw_close() { return mode("raw_close_fail") ? Status::MAXLAB_API_FAIL : Status::MAXLAB_OK; }
inline Status DataStreamerRaw_receiveNextFrame(RawFrameData* data) {
    static uint64_t i=0; static float amplitudes[1024]{};
    static auto started=std::chrono::steady_clock::now();
    if (mode("startup_stall") && std::chrono::steady_clock::now()-started<std::chrono::milliseconds(350)) return Status::MAXLAB_NO_FRAME;
    if (i>=2500) return Status::MAXLAB_NO_FRAME;
    data->frameInfo={100000+i, 0, mode("corrupt") && i==500};
    if (mode("raw_gap") && i>=500) data->frameInfo.frame_number+=1;
    if (mode("raw_reset") && i==500) data->frameInfo.frame_number=1;
    data->amplitudes=amplitudes; ++i; return Status::MAXLAB_OK;
}
inline Status DataStreamerFiltered_receiveNextFrame(FilteredFrameData* data) {
    static uint64_t filtered_index=0;static SpikeEvent framed_event{};
    if (mode("filtered_progress") || mode("filtered_gap") || mode("filtered_corrupt") || mode("event_frame_mismatch")) {
        if (filtered_index>=2500) return Status::MAXLAB_NO_FRAME;
        uint64_t frame=100000+filtered_index;
        if (mode("filtered_gap") && filtered_index>=500) ++frame;
        data->frameInfo={frame,0,mode("filtered_corrupt") && filtered_index==500};
        data->spikeCount=filtered_index==500?1:0;
        framed_event={static_cast<unsigned long>(frame+(mode("event_frame_mismatch")?1:0)),-3.125f,42,0};
        data->spikeEvents=&framed_event;++filtered_index;return Status::MAXLAB_OK;
    }
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
