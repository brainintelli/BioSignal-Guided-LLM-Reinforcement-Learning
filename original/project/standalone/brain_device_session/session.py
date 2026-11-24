"""Bound three-site single-pulse localization with the released BrainDevice recording/stimulation backend."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
import uuid
import shutil
import traceback
from types import SimpleNamespace

HOME = Path(__file__).resolve().parent
DEVICE = HOME / "BrainDevice-ubuntu22.04-x64"
READOUT = HOME / "baseline-tools"
DATA = Path("/data/local/disk1/mea_live_pair")
REG = Path.home() / ".local/state/mea_photo_full"
os.environ["HDF5_PLUGIN_PATH"] = "/pub/soft/MaxLab/so"
sys.path[:0] = [str(DEVICE/"stimulation")] + [str(READOUT/p) for p in
    ("", "mea_stimulation_v2", "mea_online", "mea_state2_r2", "mea_binary_r2")] + [
    "/pub/soft/MaxLab/python/lib/python3.10/site-packages"]
import numpy as np
from brain_stimulation import runner, transport, maxlab as mx
from brain_stimulation.config import load_config
from livepair.device import write_once
from livepair.protocol import binding, digest
from livepair.readout import read_window
from device_recording import H5Recording
from mea_stim_v2.qc import stability
from mea_stim_v2.live import _server_snapshot
from quiet_baseline import baseline
from select_sites import spatial_choice
from device_stream import Receiver
from livepair.device import Device
from livepair.device_rpc import Session as PairedSession

def native_event(message):
    try:
        return json.loads(message)
    except json.JSONDecodeError:
        fields=re.fullmatch(r"amplitude ([0-9]+) event_id ([0-9]+)",message)
        if not fields:raise
        return dict(amplitude=fields[1],event_id=fields[2])

CFG = SimpleNamespace(threshold_h5_dataset="settings/spike_threshold",decode_event=native_event)
KNOWN_STREAMS = {"49c98caf128d0006903b910127211d51acca6830b4c8c3680a63980de143050d",
                 "2bd3826b2d7646a3607f8737d8fe1767d0637ae34fea480f844de970186124f8"}

def load(path):
    return json.loads(path.read_text(encoding="utf8"))

def stream_pid():
    lines = subprocess.check_output(["ss","-Hltnp","sport","=",":5555"],text=True).strip().splitlines()
    if not lines:
        return None
    if len(lines)!=1:
        raise RuntimeError("Ambiguous 5555 listener")
    match=re.search(r"pid=(\d+)",lines[0])
    if not match:
        raise RuntimeError("Cannot identify 5555 listener")
    pid=int(match[1])
    if hashlib.sha256(Path(f"/proc/{pid}/exe").read_bytes()).hexdigest() not in KNOWN_STREAMS:
        raise RuntimeError("Unknown 5555 listener")
    return pid

def start_stream(folder):
    pid=stream_pid()
    if pid:
        return pid
    log=folder/"acquisition.log"
    with log.open("xb") as f:
        proc=subprocess.Popen([str(DEVICE/"run.sh"),"acquisition","5555"],stdout=f,stderr=f,
            stdin=subprocess.DEVNULL,start_new_session=True,env=dict(os.environ,BRAIN_DEVICE_QUIET_SPIKES="1",
            BRAIN_DESKTOP_KEEP_TERMINAL="0",PYTHONUNBUFFERED="1"))
    try:
        for _ in range(150):
            if proc.poll() is not None:raise RuntimeError("5555 acquisition exited")
            if "BrainDevice acquisition ready" in log.read_text(errors="replace") and stream_pid():
                return stream_pid()
            time.sleep(.2)
        raise RuntimeError("5555 SDK readiness timed out")
    except BaseException:
        if proc.poll() is None:os.killpg(proc.pid,signal.SIGTERM)
        raise

def params(electrodes,site,route):
    return dict(pulses=1,ipi=10,pulse_trains=1,ipti=1,amplitude_range=[200,200,1],randomize=False,
        control=False,phase=200,electrodes_selection=dict(recording=electrodes,stimulation=[site]),
        pre_stim_period=.25,post_stim_period=1.25,record_only_spikes=False,assay_path=str(route))

def map_of(text):
    first=text.split(None,1)
    if not first:return {}
    return {int(c):int(e) for c,e in re.findall(r"(\d+)\((\d+)\)",first[0])}

def save_geometry(folder,plan):
    sites=plan["suggested"]
    x=[sites[k]["x_um"] for k in "HWF"];y=[sites[k]["y_um"] for k in "HWF"]
    scale=min(415/max(1,max(x)-min(x)),235/max(1,max(y)-min(y)))
    positions={k:(42+(x[i]-min(x))*scale,55+(y[i]-min(y))*scale) for i,k in enumerate("HWF")}
    parts=['<svg xmlns="http://www.w3.org/2000/svg" width="760" height="355" viewBox="0 0 760 355">',
           '<rect width="760" height="355" fill="white"/><g font-family="sans-serif" fill="#19313a">',
           '<text x="22" y="25" font-size="19">H / W / F · electrode map</text>']
    for pair,distance in plan["distances_um"].items():
        first,second=pair.split("-");a,b=positions[first];c,d=positions[second]
        parts.append(f'<path d="M{a:.1f},{b:.1f} L{c:.1f},{d:.1f}" stroke="#769098" fill="none"/>')
    for i,(k,(a,b)) in enumerate(positions.items()):
        row=sites[k];color=("#a23d2d","#196896","#277d52")[i]
        parts.append(f'<circle cx="{a:.1f}" cy="{b:.1f}" r="6" fill="{color}"/><text x="{a+9:.1f}" y="{b+5:.1f}">{k}</text>')
        parts.append(f'<text x="510" y="{62+i*29}">{k}  #{row["electrode"]}  ({x[i]:g}, {y[i]:g}) µm</text>')
    for i,(pair,dist) in enumerate(plan["distances_um"].items()):
        parts.append(f'<text x="510" y="{178+i*29}">{pair}  {dist:.1f} µm</text>')
    parts.append('</g></svg>')
    (folder/"HWF-sites.svg").write_text("".join(parts),encoding="utf8")

def software_plan(rec,selection,folder,endpoint):
    """One software array, detached from hardware. Freeze the mapping before constructing the JSONs."""
    all_e=list(dict(rec.mapping).values())
    active_ch=set(map(int,rec.spike_channels))
    active=[e for c,e in rec.mapping if c in active_ch]
    priority=[selection["suggested"][s]["electrode"] for s in "HWF"]
    candidates=selection["candidates"]
    with transport.bind_sdk(mx,endpoint):
        online=mx.send_raw("mea_array_command online\nmea_array_get_config")
        for strategy in ("as-recorded","active-and-distributed","active"):
            a=mx.Array("localize_"+uuid.uuid4().hex)
            try:
                if strategy=="as-recorded":
                    if map_of(online)!=dict(rec.mapping):continue
                    a.load_config_data(online)
                else:
                    pool=set(active)|set(priority)
                    if strategy=="active-and-distributed":pool.update(all_e[::5])
                    a.reset();a.clear_selected_electrodes()
                    a.select_electrodes(sorted(pool));a.select_stimulation_electrodes(priority);a.route()
                for unit in range(32):
                    amp=a.query_amplifier_at_stimulation(unit).strip()
                    if amp not in ("","-1"):a.disconnect_amplifier_from_stimulation(int(amp))
                current=map_of(mx.send_raw("mea_array_command "+a.token+"\nmea_array_get_config"))
                if len(current)<4:continue
                verified=[]
                for row in candidates:
                    e=row["electrode"]
                    amp=a.query_amplifier_at_electrode(e).strip()
                    if not amp.isdigit() or current.get(int(amp))!=e:continue
                    a.connect_electrode_to_stimulation(e)
                    unit=a.query_stimulation_at_electrode(e).strip()
                    if unit.isdigit() and 0<=int(unit)<32 and a.query_amplifier_at_stimulation(int(unit)).strip()==amp:
                        verified.append(dict(row,channel=int(amp),unit=int(unit)))
                    a.disconnect_amplifier_from_stimulation(int(amp))
                if len(verified)<3:continue
                # Each point is run separately; a shared unit across runs is valid.
                chosen=spatial_choice(verified,np.asarray(selection["quadrant_midpoint_um"]))
                joint={}
                for label,row in chosen["suggested"].items():
                    e=row["electrode"]
                    a.connect_electrode_to_stimulation(e)
                    unit=a.query_stimulation_at_electrode(e).strip()
                    if unit.isdigit() and a.query_amplifier_at_stimulation(int(unit)).strip()==str(row["channel"]):
                        joint[label]=int(unit)
                for row in chosen["suggested"].values():a.disconnect_amplifier_from_stimulation(row["channel"])
                route=folder/"recording.cfg"
                a.save_config(str(route))
                if map_of(route.read_text())!=current:raise ValueError("Frozen routing map changed")
                return dict(**chosen,strategy=strategy,route=str(route),
                    route_sha256=hashlib.sha256(route.read_bytes()).hexdigest(),
                    mapping=sorted(current.items()),hardware_downloaded=False,joint_units=joint)
            finally:a.close()
    raise RuntimeError("No software route with three confirmed active sites")

def configs(plan,folder):
    result={}
    for label,row in plan["suggested"].items():
        path=folder/(label+".json")
        write_once(path,params([e for _,e in plan["mapping"]], row["electrode"],plan["route"]))
        result[label]=load_config(path)
    return result

def available_time(start,wall,mono_deadline,reserve=0):
    if wall<start or wall+reserve>=start+5700 or time.monotonic()+reserve>=mono_deadline:
        raise TimeoutError("Original sample stimulation time exhausted")

def postcheck(path,b,plan,label,dac_bits):
    row=plan["suggested"][label]
    with H5Recording(path,CFG) as rec:
        if rec.metadata["chip_id"]!=b["chip_id"] or rec.fs!=20000 or rec.mapping!=[tuple(r) for r in plan["mapping"]]:
            raise ValueError("Physical chip or actual recorded mapping differs")
        if len(rec.events)!=1:raise ValueError("Expected one hardware event")
        e=rec.events[0]
        if (e["id"]!=2 or e["eventtype"]!=1 or str(e["message"].get("amplitude"))!=str(dac_bits)
                or str(e["message"].get("event_id"))!="2"):raise ValueError("Unexpected pulse marker or DAC code")
        t=e["time_s"];channel=row["channel"]
        med=float(np.median(rec.sample([channel],t-3)))
        raw=rec.sample([channel],t-.003,.01)[0].astype(float)
        mv=np.abs(raw-med)*rec.metadata["lsb_uv"]*.001
        onset=bool(max(mv[52:60])<1 and max(mv[60:100])>=1)
        late=rec.sample([channel],t+12)
        recovered=bool(np.all((late>0)&(late<1023)) and abs(np.median(late)-med)<=5)
        exclude={rr["channel"] for rr in plan["suggested"].values()}
        pre=(rec.spike_times>=t-10)&(rec.spike_times<t-1)
        roi=sorted(set(map(int,rec.spike_channels[pre]))-exclude)
        out=dict(chip=b["chip_id"],site=label,electrode=row["electrode"],channel=channel,unit=row["unit"],
            source=rec.source,event_frame=e["frame"],onset_visible=onset,stim_channel_recovered_at_12s=recovered,
            roi_channels=roi,training_admitted=False,raw_peak_adc_mv=float(max(mv[60:100])))
        if len(roi)>=2:
            out["late_state2"]=read_window(rec,e["frame"]+240000,roi,digest(b),label)
            out["independent_recovery"]=[stability(rec.sample(roi,t-3),rec.sample(roi,t+off))["fraction_channels_passed"] for off in (2,10,12)]
        return out

def run_point(label,params_for_site,folder,plan,b,endpoint,acq_pid,start,mono_deadline,wire):
    dest=folder/label
    file_checked=False
    waits=iter(("download","before-pulse","pulse","after-pulse"))
    def guarded_wait(seconds):
        nonlocal file_checked
        stage=next(waits)
        expected={"download":mx.Timing.waitAfterDownload,"before-pulse":params_for_site.pre_recording_seconds(),
                  "pulse":params_for_site.stimulation_seconds(),"after-pulse":params_for_site.post_recording_seconds()}
        if seconds!=expected[stage] or (stage in ("pulse","after-pulse") and not file_checked):
            raise RuntimeError("Unexpected native recording/sequence order")
        h5=None
        if stage=="before-pulse":
            reports=list(dest.glob("*/run.json"))
            if len(reports)!=1:raise RuntimeError("Missing or ambiguous recording status before pulse")
            h5=Path(load(reports[0])["h5_directory_on_server"])/"stimulation.raw.h5"
        first_size=None
        until=time.monotonic()+seconds
        while time.monotonic()<until:
            if stream_pid()!=acq_pid:raise RuntimeError("5555 acquisition changed")
            if h5 is not None and first_size is None and h5.exists():
                size=h5.stat().st_size
                if size>65536:first_size=size
            time.sleep(min(.5,max(0,until-time.monotonic())))
        reports=list(dest.glob("*/run.json"))
        if stage!="before-pulse":return
        if len(reports)!=1:raise RuntimeError("Missing or ambiguous recording status before pulse")
        status=load(reports[0])
        if status["status"]!="recording":raise RuntimeError("Native recorder not ready before pulse")
        # The released runner calls this pre-recording wait directly before sequence.send().
        h5=Path(status["h5_directory_on_server"])/"stimulation.raw.h5"
        if not h5.is_file() or str(h5) not in _server_snapshot()[1]:raise RuntimeError("Bound H5 not open")
        if first_size is None:first_size=h5.stat().st_size
        for i in range(5):
            if h5.stat().st_size>max(first_size,65536):break
            if i==4:raise RuntimeError("Bound H5 has no incoming raw data")
            time.sleep(.75)
        row=plan["suggested"][label]
        if (status["stimulation_units"]!={str(row["electrode"]):row["unit"]} or
                map_of((reports[0].parent/"routing.cfg").read_text())!=dict(plan["mapping"])):
            raise RuntimeError("Actual site or downloaded recording configuration differs")
        if stream_pid()!=acq_pid:raise RuntimeError("5555 acquisition exited")
        if wire.error or wire.count==0:raise RuntimeError("5555 has no usable return")
        available_time(start,time.time(),mono_deadline)
        nominal=status["actual_amplitude_schedule_mv"]
        if len(nominal)!=1 or not 0<nominal[0]<=200:raise RuntimeError("Unexpected DAC schedule")
        write_once(folder/("attempt-"+label+".json"),dict(time=time.time(),site=label,pulses_reserved=1,
                   nominal_amplitude_mv=nominal[0],unit=row["unit"],electrode=row["electrode"]))
        file_checked=True
    code=runner.run_experiment(params_for_site,endpoint,dest,wait=guarded_wait)
    if code or not file_checked:raise RuntimeError("BrainDevice point failed: "+label)
    reports=list(dest.glob("*/run.json"));status=load(reports[0])
    if status["status"]!="completed" or status["cleanup_errors"] or _server_snapshot()[1]:
        raise RuntimeError("Recording close incomplete")
    result=postcheck(Path(status["h5_directory_on_server"])/"stimulation.raw.h5",b,plan,label,
                     round(status["actual_amplitude_schedule_mv"][0]/status["dac_lsb_mv"]))
    write_once(folder/(label+"-result.json"),result)
    if not result["stim_channel_recovered_at_12s"]:
        raise RuntimeError("Stimulation channel recovery did not verify: "+label)

def localize(b):
    if b["record_root"]!=str(DATA) or b["device_hostname"]!="researcher-biolab-1" or b["acquisition_host"]!="192.0.2.21":
        raise ValueError("Invalid acquisition binding")
    start=binding(b);deadline=time.monotonic()+start+5700-time.time()
    available_time(start,time.time(),deadline,420)
    if socket.gethostname()!=b["device_hostname"] or b["acquisition_host"] not in subprocess.check_output(["hostname","-I"],text=True).split():
        raise RuntimeError("Wrong acquisition host")
    REG.mkdir(parents=True,exist_ok=True)
    key=REG/("pair-"+digest([b["chip_id"],b["organoid_id"]])+".json")
    if b.get("replaces_session"):
        previous=load(key)
        prior=previous["binding"]
        directory=Path(previous["output"])
        if (prior["session_id"]!=b["replaces_session"] or directory!=DATA/prior["session_id"] or
                (prior["chip_id"],prior["organoid_id"])!=(b["chip_id"],b["organoid_id"])):
            raise ValueError("Previous registration differs from replacement reference")
        closed=load(directory/"closed.json")
        if closed["errors"] or not binding(prior)<=closed["time"]<start:
            raise ValueError("Prior sample did not close before the replacement")
        key=REG/("replacement-"+digest([b["chip_id"],b["organoid_id"],prior["session_id"]])+".json")
    if key.exists():raise RuntimeError("Sample has an existing study registration")
    if _server_snapshot()[1]:raise RuntimeError("Another recording is open")
    if shutil.disk_usage(DATA).free<383385600000:raise RuntimeError("Insufficient storage for the scheduled full study")
    if subprocess.check_output(["ss","-Htn","state","established","sport","=",":5555"],text=True).strip():
        raise RuntimeError("5555 single-client stream already in use")
    folder=DATA/b["session_id"];folder.mkdir(exist_ok=False)
    write_once(folder/"binding.json",b)
    write_once(key,dict(binding=b,output=str(folder)))
    acq=start_stream(folder)
    selection=baseline(b)
    with (REG/"device.lock").open("a") as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if _server_snapshot()[1]:raise RuntimeError("Acquisition already recording")
        endpoint=transport.Endpoint(timeout=5)
        quiet=DATA/(b["session_id"]+"-site-selection")/"quiet.raw.h5"
        with H5Recording(quiet,CFG) as rec:
            if rec.metadata["chip_id"]!=b["chip_id"]:raise ValueError("New baseline chip mismatch")
            plan=software_plan(rec,selection,folder,endpoint)
        write_once(folder/"site-plan.json",plan)
        save_geometry(folder,plan)
        parameters=configs(plan,folder)
        for value in parameters.values():runner.check_connection(value,endpoint)
        wire=Receiver(folder);wire.thread.start()
        try:
            for label in "HWF":
                available_time(start,time.time(),deadline,32)
                if _server_snapshot()[1]:raise RuntimeError("Another H5 is open")
                run_point(label,parameters[label],folder,plan,b,endpoint,acq,start,deadline,wire)
        finally:
            wire.stop.set();wire.thread.join(4)
            write_once(folder/"session-end.json",dict(time=time.time(),attempts_reserved=len(list(folder.glob("attempt-*.json"))),
                completed_sites=[s for s in "HWF" if (folder/(s+"-result.json")).exists()],stream_pid=stream_pid(),
                cpp_spikes=wire.count,cpp_error=wire.error,open_h5=_server_snapshot()[1]))
    return folder,plan,selection

class Study(PairedSession):
    def __init__(self):
        super().__init__()
        self.wire=None
        self.lease=None

    def open(self,b):
        if self.dev or b.get("device_flow")!="braindevice_5555_7215_v1":raise ValueError("Invalid study registration")
        start=binding(b)
        if not 0<=time.time()-start<5700:raise TimeoutError("Outside original stimulation window")
        signal.setitimer(signal.ITIMER_REAL,max(.01,start+7200-time.time()-120))
        folder,plan,selection=localize(b)
        self.lease=(REG/"device.lock").open("a")
        fcntl.flock(self.lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if _server_snapshot()[1]:raise RuntimeError("Another recording is open")
        if not all(load(folder/(s+"-result.json"))["onset_visible"] for s in "HWF"):
            raise RuntimeError("Localization complete; at least one site has no verified onset")
        if set(plan["joint_units"])!=set("HWF") or len(set(plan["joint_units"].values()))!=3:
            raise RuntimeError("Single-point tests completed; no three independent simultaneous routes")
        study=dict(b,material="Pt-Black",electrodes={k:r["electrode"] for k,r in plan["suggested"].items()},
                   route_file=plan["route"],route_sha256=plan["route_sha256"],prior_nonzero_attempts=3)
        out=folder/"paired";out.mkdir()
        write_once(out/"binding.json",study)
        write_once(out/"spatial-selection.json",dict(selection,**{k:plan[k] for k in
            ("suggested","distances_um","minimum_distance_um","region_count","quadrant_midpoint_um")}))
        self.wire=Receiver(out);self.wire.thread.start()
        d=self.dev=Device(study,out);d.selected=plan["suggested"]
        d.lsb=d.preflight()["lsb_mv"];d.prepare_sites()
        if d.units!=plan["joint_units"]:raise RuntimeError("Joint route differs from software readback")
        windows=d.prime("baseline",True)
        original=d.before_sequence_send
        def reserve(event):
            if self.wire.error or self.wire.count==0:raise RuntimeError("No usable TCP stream for paired recording")
            original(event)
        d.before_sequence_send=reserve
        return dict(binding=d.bid,resolved_binding=study,windows=windows,norm=d.norm,out=str(out))

    def close(self):
        if hasattr(self,"close_result"):return self.close_result
        errors=self.dev.cleanup() if self.dev else []
        if self.wire:
            self.wire.stop.set();self.wire.thread.join(4)
        if self.lease:self.lease.close()
        if self.dev:write_once(self.dev.run_dir/"closed.json",dict(errors=errors,attempts=self.dev.attempts,time=time.time()))
        self.close_result=dict(closed=not errors,errors=errors)
        return self.close_result

def expiry(*_):raise TimeoutError("Session deadline or disconnect")

def main():
    for sig in (signal.SIGALRM,signal.SIGHUP,signal.SIGTERM):signal.signal(sig,expiry)
    if sys.argv[1:]==["device"]:
        out=os.fdopen(os.dup(1),"w",encoding="utf8",buffering=1)
        os.dup2(sys.stderr.fileno(),1)
        study=Study()
        try:
            for line in sys.stdin:
                cmd=json.loads(line);op=cmd["op"]
                if op=="close":break
                if op not in ("protocol","open","calibrate","prime","slot"):raise ValueError("Unknown operation")
                try:value=getattr(study,op)(**cmd.get("args",{}))
                except Exception:
                    error=traceback.format_exc()
                    study.close()
                    print(json.dumps(dict(ok=False,error=error)),file=out,flush=True)
                    # The workstation collects any finished files, then requests the close receipt.
                    sys.stdin.readline()
                    break
                print(json.dumps(dict(ok=True,value=value),allow_nan=False),file=out,flush=True)
        finally:
            result=study.close()
            print(json.dumps(dict(ok=result["closed"],value=result)),file=out,flush=True)
        return
    parser=argparse.ArgumentParser(description="BrainDevice engineering localization only")
    parser.add_argument("--chip",required=True);parser.add_argument("--organoid",required=True)
    parser.add_argument("--started-at",required=True)
    args=parser.parse_args()
    b=dict(session_id="bd-"+uuid.uuid4().hex[:16],replicate=0,acquisition_host="192.0.2.21",
           device_hostname="researcher-biolab-1",device_model="MaxOne",chip_id=args.chip,organoid_id=args.organoid,
           started_at=args.started_at,record_root=str(DATA),bad_channels=[],threshold_value=5,threshold_h5_dataset=CFG.threshold_h5_dataset)
    start=binding(b);available_time(start,time.time(),time.monotonic()+start+5700-time.time())
    signal.setitimer(signal.ITIMER_REAL,max(.01,start+7200-time.time()-120))
    folder,_,_=localize(b)
    print(str(folder))

if __name__=="__main__":main()
