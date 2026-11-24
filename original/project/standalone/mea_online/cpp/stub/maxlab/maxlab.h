// Explicit test-only SDK. Never link this source into a live collector.
#pragma once
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <thread>
#define MAXLAB_PUBLIC_API_VERSION "simulation-stub"
#ifndef MEA_SIMULATION
#error "Fake SDK requires explicit MEA_SIMULATION build label"
#endif
namespace maxlab {
enum class Status { MAXLAB_OK=0, MAXLAB_NO_FRAME=1, MAXLAB_ERROR=2 };
enum class FilterType { FIR=0 };
struct FrameInfo { uint64_t frame_number=0; uint8_t well_id=0; bool corrupted=false; };
struct SpikeEvent { uint64_t frameNo=0; uint16_t channel=0; uint8_t wellId=0; float amp=0; };
struct FilteredFrameData { uint64_t spikeCount=0; const SpikeEvent* spikeEvents=nullptr; FrameInfo frameInfo; };
inline const char* getStaticLibraryVersion() { return MAXLAB_PUBLIC_API_VERSION; }
inline bool mode(const char* value) { const char* current=std::getenv("MEA_STUB_MODE"); return current && std::strcmp(current,value)==0; }
inline uint64_t count=0;
inline std::chrono::steady_clock::time_point opened;
inline Status DataStreamerFiltered_open(FilterType) {
    count=0; opened=std::chrono::steady_clock::now();
    return mode("open_error") ? Status::MAXLAB_ERROR : Status::MAXLAB_OK;
}
inline Status DataStreamerFiltered_close() { return mode("close_error") ? Status::MAXLAB_ERROR : Status::MAXLAB_OK; }
inline Status DataStreamerFiltered_receiveNextFrame(FilteredFrameData* value) {
    if (mode("timeout")) { std::this_thread::sleep_for(std::chrono::milliseconds(1)); return Status::MAXLAB_NO_FRAME; }
    if (mode("sdk_error")) return Status::MAXLAB_ERROR;
    if (!mode("backlog") && !(mode("too_fast") && count>=20000))
        std::this_thread::sleep_until(opened+std::chrono::microseconds(50*(count+1)));
    value->frameInfo={1000+count,0,false};
    if (count==20005) {
        if (mode("gap")) ++value->frameInfo.frame_number;
        if (mode("reorder")) value->frameInfo.frame_number-=2;
        if (mode("corrupt")) value->frameInfo.corrupted=true;
        if (mode("well")) value->frameInfo.well_id=1;
    }
    static SpikeEvent events[2];
    events[0]={value->frameInfo.frame_number-10,0,0,-2.5F};
    value->spikeCount=(!mode("quiet") && count%20000==0) ? 1 : 0;
    value->spikeEvents=events;
    if (count==20000) {
        if (mode("event_well")) events[0].wellId=1;
        if (mode("future_peak")) events[0].frameNo=value->frameInfo.frame_number+1;
        if (mode("unknown_channel")) events[0].channel=3;
        if (mode("duplicate")) { events[1]=events[0]; value->spikeCount=2; }
    }
    ++count;
    return Status::MAXLAB_OK;
}
}
