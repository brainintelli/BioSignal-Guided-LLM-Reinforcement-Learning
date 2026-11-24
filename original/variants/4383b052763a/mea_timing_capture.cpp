// Bounded diagnostic: raw coverage is never a filtered-event watermark.
#include "maxlab/maxlab.h"
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <string>
#include <time.h>
#include <unistd.h>

using U = unsigned long long;
static volatile sig_atomic_t stopped = 0;
static void stop(int) { stopped = 1; }
static U clock_ns(clockid_t id) { timespec t{}; clock_gettime(id, &t); return U(t.tv_sec)*1000000000ULL+U(t.tv_nsec); }
static U mono() { return clock_ns(CLOCK_MONOTONIC_RAW); }
static FILE* exclusive(const char* name) {
    int fd = open(name,O_WRONLY|O_CREAT|O_EXCL,0600);
    if (fd<0) return nullptr;
    FILE* f=fdopen(fd,"w"); if (!f) close(fd); return f;
}

int main(int argc,char** argv) {
    if (argc!=3) return 64;
    char* tail=nullptr; long seconds=std::strtol(argv[2],&tail,10);
    if (!tail || *tail || seconds<1 || seconds>120) return 64;
    FILE* events=exclusive(argv[1]); if (!events) return 65;
    std::string progress_path=std::string(argv[1])+".progress.jsonl";
    FILE* progress=exclusive(progress_path.c_str());
    if (!progress) { std::fclose(events); return 65; }
    std::signal(SIGINT,stop); std::signal(SIGTERM,stop);
    int filtered_open=int(maxlab::DataStreamerFiltered_open(maxlab::FilterType::FIR));
    int raw_open=filtered_open==0 ? int(maxlab::DataStreamerRaw_open()) : -1;
    int opened=filtered_open ? filtered_open : raw_open;
    std::fprintf(events,"{\"type\":\"session\",\"seq\":0,\"protocol\":\"mea.filtered.p1.v1\",\"sdk_open_status\":%d,\"raw_open_status\":%d,\"filtered_open_status\":%d}\n",opened,raw_open,filtered_open);
    std::fprintf(progress,"{\"type\":\"session\",\"seq\":0,\"protocol\":\"mea.raw_progress.v1\",\"raw_open_status\":%d,\"filtered_open_status\":%d}\n",raw_open,filtered_open);
    bool io_ok=std::fflush(events)==0 && std::fflush(progress)==0;
    U es=1,ps=1,event_count=0,raw_count=0,filtered_count=0,block_count=0,block_start=0,last_frame=0;
    U last_begin=0,last_end=0,last_progress=mono(); int errors=0;
    auto block=[&]() {
        if (!block_count) return;
        U before=mono(), wall=clock_ns(CLOCK_REALTIME), after=mono();
        std::fprintf(progress,"{\"type\":\"raw_block\",\"seq\":%llu,\"start_frame\":%llu,\"end_frame\":%llu,\"count\":%llu,\"cumulative_frames\":%llu,\"recv_begin_ns\":%llu,\"recv_end_ns\":%llu,\"wall_before_ns\":%llu,\"wall_after_ns\":%llu,\"wall_ns\":%llu,\"filtered_frames\":%llu,\"filtered_events\":%llu}\n",
                     ps++,block_start,last_frame+1,block_count,raw_count,last_begin,last_end,before,after,wall,filtered_count,event_count);
        io_ok=std::fflush(progress)==0 && io_ok; block_count=0; last_progress=after;
    };
    auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(seconds);
    while (opened==0 && io_ok && !stopped && std::chrono::steady_clock::now()<deadline) {
        bool received=false;
        maxlab::RawFrameData raw{};
        U begin=mono(); auto rs=maxlab::DataStreamerRaw_receiveNextFrame(&raw); U end=mono();
        if (rs==maxlab::Status::MAXLAB_OK) {
            received=true;
            U frame=raw.frameInfo.frame_number;
            if (raw.frameInfo.corrupted || raw.frameInfo.well_id!=0 || (raw_count && frame!=last_frame+1) || raw_count>=5000000) {
                std::fprintf(progress,"{\"type\":\"fault\",\"seq\":%llu,\"frame\":%llu,\"previous_frame\":%llu,\"corrupted\":%s,\"well_id\":%u}\n",
                             ps++,frame,last_frame,raw.frameInfo.corrupted?"true":"false",unsigned(raw.frameInfo.well_id));
                ++errors; break;
            }
            if (!block_count) block_start=frame;
            ++block_count; ++raw_count; last_frame=frame; last_begin=begin; last_end=end;
            if (block_count==1000) block();
        } else if (rs!=maxlab::Status::MAXLAB_NO_FRAME) { ++errors; break; }
        maxlab::FilteredFrameData data{};
        begin=mono(); auto fs=maxlab::DataStreamerFiltered_receiveNextFrame(&data); end=mono();
        if (fs==maxlab::Status::MAXLAB_OK) {
            received=true; ++filtered_count;
            if (data.spikeCount>1024 || (data.spikeCount && !data.spikeEvents) || event_count+data.spikeCount>5000000 || filtered_count>5000000) { ++errors; break; }
            bool valid=true;
            for (uint64_t i=0;i<data.spikeCount;++i) valid &= std::isfinite(data.spikeEvents[i].amp);
            if (!valid) { ++errors; break; }
            if (data.spikeCount) {
                std::fprintf(events,"{\"type\":\"events\",\"seq\":%llu,\"recv_begin_ns\":%llu,\"recv_end_ns\":%llu,\"latest_raw_end_frame\":%llu,\"events\":[",es++,begin,end,raw_count?last_frame+1:0);
                for (uint64_t i=0;i<data.spikeCount;++i) {
                    const auto& s=data.spikeEvents[i];
                    std::fprintf(events,"%s[%lu,%u,%u,%.9g]",i?",":"",s.frameNo,unsigned(s.channel),unsigned(s.wellId),s.amp);
                }
                std::fprintf(events,"]}\n"); event_count+=data.spikeCount;
                io_ok=std::fflush(events)==0 && io_ok;
            }
        } else if (fs!=maxlab::Status::MAXLAB_NO_FRAME) { ++errors; break; }
        U now=mono();
        if (now-last_progress>=250000000ULL) {
            std::fprintf(progress,"{\"type\":\"heartbeat\",\"seq\":%llu,\"raw_frames\":%llu,\"since_raw_ns\":%llu,\"mono_ns\":%llu}\n",ps++,raw_count,last_end?now-last_end:0,now);
            io_ok=std::fflush(progress)==0 && io_ok; last_progress=now;
        }
        if (!received) usleep(100);
    }
    block();
    int raw_close=raw_open==0 ? int(maxlab::DataStreamerRaw_close()) : -1;
    int filtered_close=filtered_open==0 ? int(maxlab::DataStreamerFiltered_close()) : -1;
    int close_status=(raw_open==0 && raw_close!=0) ? raw_close : ((filtered_open==0 && filtered_close!=0) ? filtered_close : 0);
    bool clean=opened==0 && close_status==0 && errors==0 && io_ok && raw_count>0;
    std::fprintf(events,"{\"type\":\"end\",\"seq\":%llu,\"clean_end\":%s,\"sdk_close_status\":%d,\"sdk_errors\":%d,\"event_count\":%llu,\"raw_close_status\":%d,\"filtered_close_status\":%d,\"received_frames\":%llu,\"upstream_loss\":null}\n",
                 es,clean?"true":"false",close_status,errors,event_count,raw_close,filtered_close,filtered_count);
    std::fprintf(progress,"{\"type\":\"end\",\"seq\":%llu,\"raw_frames\":%llu,\"raw_contiguous\":%s,\"clean_end\":%s,\"raw_close_status\":%d,\"filtered_close_status\":%d,\"errors\":%d}\n",
                 ps,raw_count,(errors==0 && raw_count>0)?"true":"false",clean?"true":"false",raw_close,filtered_close,errors);
    if (std::fflush(events)!=0 || fsync(fileno(events))!=0) clean=false;
    if (std::fflush(progress)!=0 || fsync(fileno(progress))!=0) clean=false;
    if (std::fclose(events)!=0) clean=false;
    if (std::fclose(progress)!=0) clean=false;
    return clean?0:2;
}
