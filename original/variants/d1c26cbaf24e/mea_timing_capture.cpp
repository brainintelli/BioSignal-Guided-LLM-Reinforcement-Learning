// Bounded diagnostic: raw coverage is never a filtered-event watermark.
#include "maxlab/maxlab.h"
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
#ifdef MAXLAB_PUBLIC_API_VERSION
    if (argc==2 && std::strcmp(argv[1],"--version")==0) {
        std::printf("headers=%s library=%s\n",MAXLAB_PUBLIC_API_VERSION,maxlab::getStaticLibraryVersion());
        return std::strcmp(MAXLAB_PUBLIC_API_VERSION,maxlab::getStaticLibraryVersion())==0?0:1;
    }
#endif
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
    U started=mono();
#ifdef MEA_FILTERED_FRAME_INFO
    const char* protocol="mea.filtered.frames.v1";
#else
    const char* protocol="mea.filtered.p1.v1";
#endif
    std::fprintf(events,"{\"type\":\"session\",\"seq\":0,\"protocol\":\"%s\",\"session_id\":\"%ld-%llu\",\"sdk_open_status\":%d,\"raw_open_status\":%d,\"filtered_open_status\":%d,\"mono_ns\":%llu}\n",protocol,long(getpid()),started,opened,raw_open,filtered_open,started);
    std::fprintf(progress,"{\"type\":\"session\",\"seq\":0,\"protocol\":\"mea.raw_progress.v1\",\"raw_open_status\":%d,\"filtered_open_status\":%d,\"mono_ns\":%llu}\n",raw_open,filtered_open,started);
    bool io_ok=std::fflush(events)==0 && std::fflush(progress)==0;
    U es=1,ps=1,event_count=0,raw_count=0,filtered_count=0,block_count=0,block_start=0,last_frame=0;
    U last_begin=0,last_end=0,first_receive=0,last_progress=started; int errors=0;
    U raw_no_frame=0,filtered_no_frame=0;
#ifdef MEA_FILTERED_FRAME_INFO
    U fb_count=0,fb_start=0,ff_last=0,ff_first_recv=0,ff_begin=0,ff_end=0,last_filtered_progress=started;
    auto filtered_block=[&]() {
        if (!fb_count) return;
        U before=mono(),wall=clock_ns(CLOCK_REALTIME),after=mono();
        std::fprintf(events,"{\"type\":\"frame_progress\",\"seq\":%llu,\"start_frame\":%llu,\"end_frame\":%llu,\"count\":%llu,\"cumulative_frames\":%llu,\"event_count\":%llu,\"complete\":true,\"discontinuities\":0,\"corrupted\":0,\"recv_begin_ns\":%llu,\"recv_end_ns\":%llu,\"wall_before_ns\":%llu,\"wall_after_ns\":%llu,\"wall_ns\":%llu,\"first_recv_ns\":%llu}\n",
                     es++,fb_start,ff_last+1,fb_count,filtered_count,event_count,ff_begin,ff_end,before,after,wall,ff_first_recv);
        io_ok=std::fflush(events)==0 && io_ok;fb_count=0;last_filtered_progress=after;
    };
#endif
    auto block=[&]() {
        if (!block_count) return;
        U before=mono(), wall=clock_ns(CLOCK_REALTIME), after=mono();
        std::fprintf(progress,"{\"type\":\"raw_block\",\"seq\":%llu,\"start_frame\":%llu,\"end_frame\":%llu,\"count\":%llu,\"cumulative_frames\":%llu,\"recv_begin_ns\":%llu,\"recv_end_ns\":%llu,\"wall_before_ns\":%llu,\"wall_after_ns\":%llu,\"wall_ns\":%llu,\"filtered_frames\":%llu,\"filtered_events\":%llu,\"first_recv_ns\":%llu}\n",
                     ps++,block_start,last_frame+1,block_count,raw_count,last_begin,last_end,before,after,wall,filtered_count,event_count,first_receive);
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
            if (!raw_count) first_receive=end;
            ++block_count; ++raw_count; last_frame=frame; last_begin=begin; last_end=end;
            if (block_count==1000) block();
        } else if (rs!=maxlab::Status::MAXLAB_NO_FRAME) { ++errors; break; }
        else { ++raw_no_frame; }
        maxlab::FilteredFrameData data{};
        begin=mono(); auto fs=maxlab::DataStreamerFiltered_receiveNextFrame(&data); end=mono();
        if (fs==maxlab::Status::MAXLAB_OK) {
            received=true;
#ifdef MEA_FILTERED_FRAME_INFO
            U frame=data.frameInfo.frame_number;
            if (data.frameInfo.corrupted || data.frameInfo.well_id!=0 || (filtered_count && frame!=ff_last+1)) { ++errors;break; }
            if (!fb_count) fb_start=frame;
            if (!filtered_count) ff_first_recv=end;
            ff_last=frame;ff_begin=begin;ff_end=end;
#endif
            ++filtered_count;
            if (data.spikeCount>1024 || (data.spikeCount && !data.spikeEvents) || event_count+data.spikeCount>5000000 || filtered_count>5000000) { ++errors; break; }
            bool valid=true;
            for (uint64_t i=0;i<data.spikeCount;++i) valid &= std::isfinite(data.spikeEvents[i].amp);
#ifdef MEA_FILTERED_FRAME_INFO
            for (uint64_t i=0;i<data.spikeCount;++i)
                if (data.spikeEvents[i].frameNo!=frame || data.spikeEvents[i].wellId!=data.frameInfo.well_id) {
                    std::fprintf(stderr,"frame association: transport=%llu peak=%lu transport_well=%u event_well=%u amplitude=%.9g\n",frame,data.spikeEvents[i].frameNo,unsigned(data.frameInfo.well_id),unsigned(data.spikeEvents[i].wellId),data.spikeEvents[i].amp);
                    valid=false;
                }
#endif
            if (!valid) { ++errors; break; }
            if (data.spikeCount) {
                std::fprintf(events,"{\"type\":\"events\",\"seq\":%llu,\"recv_begin_ns\":%llu,\"recv_end_ns\":%llu,\"latest_raw_end_frame\":%llu,",es++,begin,end,raw_count?last_frame+1:0);
#ifdef MEA_FILTERED_FRAME_INFO
                std::fprintf(events,"\"frame_number\":%llu,",frame);
#endif
                std::fprintf(events,"\"events\":[");
                for (uint64_t i=0;i<data.spikeCount;++i) {
                    const auto& s=data.spikeEvents[i];
                    std::fprintf(events,"%s[%lu,%u,%u,%.9g]",i?",":"",s.frameNo,unsigned(s.channel),unsigned(s.wellId),s.amp);
                }
                std::fprintf(events,"]}\n"); event_count+=data.spikeCount;
                io_ok=std::fflush(events)==0 && io_ok;
            }
#ifdef MEA_FILTERED_FRAME_INFO
            ++fb_count;if (fb_count==1000) filtered_block();
#endif
        } else if (fs!=maxlab::Status::MAXLAB_NO_FRAME) { ++errors; break; }
        else { ++filtered_no_frame; }
        U now=mono();
#ifdef MEA_FILTERED_FRAME_INFO
        if (now-last_filtered_progress>=250000000ULL) {
            std::fprintf(events,"{\"type\":\"frame_heartbeat\",\"seq\":%llu,\"frame_count\":%llu,\"since_frame_ns\":%llu,\"mono_ns\":%llu}\n",es++,filtered_count,now-(ff_end?ff_end:started),now);
            io_ok=std::fflush(events)==0 && io_ok;last_filtered_progress=now;
        }
#endif
        if (now-last_progress>=250000000ULL) {
            std::fprintf(progress,"{\"type\":\"heartbeat\",\"seq\":%llu,\"raw_frames\":%llu,\"since_raw_ns\":%llu,\"mono_ns\":%llu}\n",ps++,raw_count,now-(last_end?last_end:started),now);
            io_ok=std::fflush(progress)==0 && io_ok; last_progress=now;
        }
        if (!received) usleep(100);
    }
    block();
#ifdef MEA_FILTERED_FRAME_INFO
    filtered_block();
#endif
    U ended=mono();
    int raw_close=raw_open==0 ? int(maxlab::DataStreamerRaw_close()) : -1;
    int filtered_close=filtered_open==0 ? int(maxlab::DataStreamerFiltered_close()) : -1;
    int close_status=(raw_open==0 && raw_close!=0) ? raw_close : ((filtered_open==0 && filtered_close!=0) ? filtered_close : 0);
    bool clean=opened==0 && close_status==0 && errors==0 && io_ok && raw_count>0;
    std::fprintf(events,"{\"type\":\"end\",\"seq\":%llu,\"clean_end\":%s,\"sdk_close_status\":%d,\"sdk_errors\":%d,\"event_count\":%llu,\"raw_close_status\":%d,\"filtered_close_status\":%d,\"received_frames\":%llu,\"raw_no_frame\":%llu,\"filtered_no_frame\":%llu,\"end_mono_ns\":%llu,\"upstream_loss\":null}\n",
                 es,clean?"true":"false",close_status,errors,event_count,raw_close,filtered_close,filtered_count,raw_no_frame,filtered_no_frame,ended);
    std::fprintf(progress,"{\"type\":\"end\",\"seq\":%llu,\"raw_frames\":%llu,\"raw_contiguous\":%s,\"clean_end\":%s,\"raw_close_status\":%d,\"filtered_close_status\":%d,\"errors\":%d,\"end_mono_ns\":%llu}\n",
                 ps,raw_count,(errors==0 && raw_count>0)?"true":"false",clean?"true":"false",raw_close,filtered_close,errors,ended);
    if (std::fflush(events)!=0 || fsync(fileno(events))!=0) clean=false;
    if (std::fflush(progress)!=0 || fsync(fileno(progress))!=0) clean=false;
    if (std::fclose(events)!=0) clean=false;
    if (std::fclose(progress)!=0) clean=false;
    return clean?0:2;
}
