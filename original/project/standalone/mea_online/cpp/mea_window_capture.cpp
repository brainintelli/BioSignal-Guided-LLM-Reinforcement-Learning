// Request-scoped filtered transport capture. No stimulation / command API.
#include "maxlab/maxlab.h"
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef MEA_HEADER_SHA256
#error "Build requires actual SDK header SHA256 provenance"
#endif
#ifndef MEA_LIBRARY_SHA256
#error "Build requires actual SDK static library SHA256 provenance"
#endif

using Clock = std::chrono::steady_clock;
static volatile std::sig_atomic_t interrupted = 0;
static void on_signal(int) { interrupted = 1; }
static double seconds(Clock::time_point from) { return std::chrono::duration<double>(Clock::now()-from).count(); }
#ifdef MEA_SIMULATION
static constexpr const char* source = "simulation";
#else
static constexpr const char* source = "live";
#endif

static bool identifier(const std::string& value) {
    if (value.empty() || value.size()>128) return false;
    for (unsigned char ch : value)
        if (!((ch>='a' && ch<='z') || (ch>='A' && ch<='Z') || (ch>='0' && ch<='9') || ch=='_' || ch=='-' || ch=='.' || ch==':')) return false;
    return true;
}
static bool hash(const std::string& value) {
    if (value.size()!=64) return false;
    for (char ch : value) if (!((ch>='a' && ch<='f') || (ch>='0' && ch<='9'))) return false;
    return true;
}
struct Event { uint64_t transport, peak; unsigned channel; float amplitude; };

int main(int argc, char** argv) {
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);
    if (std::strcmp(MAXLAB_PUBLIC_API_VERSION, maxlab::getStaticLibraryVersion())!=0) {
        std::cerr << "SDK header/library version mismatch\n"; return 2;
    }
    // Member access intentionally fails compilation for SDKs without filtered frameInfo.
    static_assert(sizeof(maxlab::FilteredFrameData{}.frameInfo.frame_number)==sizeof(uint64_t), "filtered frameInfo required");
    if (argc==2 && std::strcmp(argv[1], "--describe-json")==0) {
        std::cout << "{\"schema\":\"mea.online.producer.v1\",\"source\":\"" << source
                  << "\",\"filtered_frame_info\":true,\"headers\":\"" << MAXLAB_PUBLIC_API_VERSION
                  << "\",\"library\":\"" << maxlab::getStaticLibraryVersion()
                  << "\",\"header_sha256\":\"" << MEA_HEADER_SHA256
                  << "\",\"library_sha256\":\"" << MEA_LIBRARY_SHA256 << "\"}\n";
        return std::cout.good() ? 0 : 3;
    }
    if (argc!=6 || std::strcmp(argv[1], "--capture")!=0 || !identifier(argv[2]) || !hash(argv[3]) || !identifier(argv[4])) {
        std::cerr << "usage: --describe-json | --capture REQUEST BINDING_SHA SESSION CHANNELS\n"; return 64;
    }
    bool opened = false;
    try {
        std::set<unsigned> channels;
        std::istringstream channel_input(argv[5]); std::string item;
        while (std::getline(channel_input, item, ',')) {
            if (item.empty() || item.find_first_not_of("0123456789")!=std::string::npos) throw std::runtime_error("channel format");
            unsigned long channel = std::stoul(item);
            if (channel>1023 || !channels.insert(unsigned(channel)).second) throw std::runtime_error("channel duplicate/range");
        }
        if (channels.empty() || channels.size()>1024) throw std::runtime_error("channels required");
        if (maxlab::DataStreamerFiltered_open(maxlab::FilterType::FIR)!=maxlab::Status::MAXLAB_OK) throw std::runtime_error("SDK open failed");
        opened = true;
        auto began=Clock::now();
        std::vector<Event> events;
        std::set<std::pair<uint64_t,unsigned>> seen;
        uint64_t last=0, start=0;
        double warmup=0, elapsed=0;
        auto phase_start=began;
        for (uint64_t index=0; index<40000; ++index) {
            if (interrupted || seconds(phase_start)>2.5 || seconds(began)>5.0) throw std::runtime_error("capture interrupted/timeout");
            maxlab::FilteredFrameData frame{};
            auto status=maxlab::DataStreamerFiltered_receiveNextFrame(&frame);
            // New SDK documents NO_FRAME as loss/error; it is NOT a drain barrier.
            if (status!=maxlab::Status::MAXLAB_OK) throw std::runtime_error("SDK receive error/NO_FRAME");
            uint64_t transport=frame.frameInfo.frame_number;
            if (frame.frameInfo.corrupted || frame.frameInfo.well_id!=0 || transport==std::numeric_limits<uint64_t>::max() ||
                (index && (last==std::numeric_limits<uint64_t>::max() || transport!=last+1))) throw std::runtime_error("filtered transport gap/order/corruption/well");
            last=transport;
            if (frame.spikeCount>1024 || (frame.spikeCount && !frame.spikeEvents)) throw std::runtime_error("spike pointer/count");
            if (index==20000) start=transport;
            for (uint64_t n=0; n<frame.spikeCount; ++n) {
                const auto& event=frame.spikeEvents[n];
                if (event.wellId!=0 || event.frameNo>transport || !std::isfinite(event.amp) || !channels.count(event.channel)) throw std::runtime_error("event well/peak/amplitude/channel");
                if (!seen.insert({event.frameNo,event.channel}).second) throw std::runtime_error("duplicate event");
                if (seen.size()>200000) throw std::runtime_error("warmup/event capacity");
                if (index>=20000) {
                    if (events.size()>=100000) throw std::runtime_error("event capacity");
                    events.push_back({transport,event.frameNo,event.channel,event.amp});
                }
            }
            if (index==19999) {
                warmup=seconds(phase_start);
                if (warmup<.75 || warmup>2.5) throw std::runtime_error("warmup accelerated backlog/timeout");
                phase_start=Clock::now();
            }
        }
        elapsed=seconds(phase_start);
        if (elapsed<.75 || elapsed>2.5) throw std::runtime_error("capture accelerated backlog/timeout");
        int closed=int(maxlab::DataStreamerFiltered_close()); opened=false;
        std::cerr << "sdk_close_status=" << closed << " warmup_frames=20000 capture_frames=20000 events=" << events.size() << "\n";
        if (closed!=0) throw std::runtime_error("SDK close failed; source inspection required before restart");
        if (interrupted) throw std::runtime_error("interrupted before publish");
        std::ostringstream out; out << std::setprecision(17);
        out << "{\"schema\":\"mea.online.window.v1\",\"source\":\"" << source
            << "\",\"request_id\":\"" << argv[2] << "\",\"binding_sha256\":\"" << argv[3]
            << "\",\"session_id\":\"" << argv[4] << "\",\"start_frame\":" << start
            << ",\"end_frame\":" << last+1 << ",\"frame_count\":20000,\"complete\":true,\"dropped\":0,\"corrupted\":0"
            << ",\"stream_opened_after_request\":true,\"warmup_frames\":20000,\"warmup_elapsed_s\":" << warmup
            << ",\"capture_elapsed_s\":" << elapsed << ",\"events\":[";
        for (size_t i=0; i<events.size(); ++i) {
            const auto& event=events[i];
            if (i) out << ',';
            out << '[' << event.transport << ',' << event.peak << ',' << event.channel << ',' << event.amplitude << ']';
        }
        out << "]}\n";
        auto result=out.str();
        if (result.size()>8*1024*1024) throw std::runtime_error("stdout capacity");
        std::cout << result;
        if (!std::cout.good()) throw std::runtime_error("stdout failed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        if (opened) std::cerr << "sdk_close_status=" << int(maxlab::DataStreamerFiltered_close()) << '\n';
        return 2;
    }
}
