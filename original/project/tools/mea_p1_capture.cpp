// File-backed filtered events. No online watermark or upstream loss claim.
#include "maxlab/maxlab.h"
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <unistd.h>

static volatile sig_atomic_t stop_requested = 0;
static void stop(int) { stop_requested = 1; }

int main(int argc, char** argv) {
    if (argc != 3) return 64;
    char* end = nullptr; long seconds = std::strtol(argv[2], &end, 10);
    if (!end || *end || seconds < 1 || seconds > 120) return 64;
    int fd = ::open(argv[1], O_WRONLY | O_CREAT | O_EXCL, 0600);
    if (fd < 0) return 65;
    FILE* out = fdopen(fd, "w"); if (!out) { ::close(fd); return 66; }
    std::signal(SIGINT, stop); std::signal(SIGTERM, stop);
    int opened = static_cast<int>(maxlab::DataStreamerFiltered_open(maxlab::FilterType::FIR));
    std::fprintf(out, "{\"type\":\"session\",\"protocol\":\"mea.filtered.p1.v1\",\"seq\":0,\"sdk_open_status\":%d}\n", opened);
    bool io_ok = std::fflush(out) == 0 && fsync(fd) == 0;
    unsigned long long seq = 1, count = 0, received_frames = 0, empty_frames = 0; int errors = 0;
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(seconds);
    while (opened == 0 && io_ok && !stop_requested && std::chrono::steady_clock::now() < deadline) {
        maxlab::FilteredFrameData data{};
        auto status = maxlab::DataStreamerFiltered_receiveNextFrame(&data);
        if (status == maxlab::Status::MAXLAB_NO_FRAME) { usleep(100); continue; }
        if (status != maxlab::Status::MAXLAB_OK || data.spikeCount > 1024 ||
            (data.spikeCount && !data.spikeEvents) || count + data.spikeCount > 5000000) {
            ++errors; break;
        }
        ++received_frames;
        // OK with zero spikes is still a sampled frame. Sleeping here throttles
        // a 20 kHz source below its sample rate and leaves a growing SDK backlog.
        if (!data.spikeCount) { ++empty_frames; continue; }
        bool valid = true;
        for (uint64_t i=0; i<data.spikeCount; ++i) valid &= std::isfinite(data.spikeEvents[i].amp);
        if (!valid) { ++errors; break; }
        std::fprintf(out, "{\"type\":\"events\",\"seq\":%llu,\"events\":[", seq++);
        for (uint64_t i=0; i<data.spikeCount; ++i) {
            const auto& s = data.spikeEvents[i];
            std::fprintf(out, "%s[%lu,%u,%u,%.9g]", i ? "," : "", s.frameNo,
                         static_cast<unsigned>(s.channel), static_cast<unsigned>(s.wellId), s.amp);
        }
        std::fprintf(out, "]}\n"); count += data.spikeCount;
        io_ok = std::fflush(out) == 0;
    }
    int closed = opened == 0 ? static_cast<int>(maxlab::DataStreamerFiltered_close()) : -1;
    bool ok = opened == 0 && closed == 0 && errors == 0 && io_ok;
    std::fprintf(out, "{\"type\":\"end\",\"seq\":%llu,\"clean_end\":%s,\"sdk_close_status\":%d,\"sdk_errors\":%d,\"event_count\":%llu,\"received_frames\":%llu,\"empty_frames\":%llu,\"upstream_loss\":null,\"stop_requested\":%s}\n",
                 seq, ok ? "true" : "false", closed, errors, count, received_frames, empty_frames, stop_requested ? "true" : "false");
    ok = std::fflush(out) == 0 && fsync(fd) == 0 && ok;
    if (std::fclose(out) != 0) ok = false;
    return ok ? 0 : 2;
}
