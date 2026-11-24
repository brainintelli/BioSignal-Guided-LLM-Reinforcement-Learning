"""Run released BrainDevice output against an ephemeral loopback instrument, never the MaxOne port."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
from datetime import datetime,timezone

import session as s
import h5py
import numpy as np
from fixture_upstream import FakeMaxLab

def make_recording(path,plan,label,chip,raw_marker):
    n=32*20000
    anchor=15*20000
    stim=plan["suggested"][label]["channel"]
    with h5py.File(path,"w") as f:
        f["wellplate/id"]=[chip.encode()]
        d=f.create_group("data_store/data0000")
        for k,v in dict(sampling=20000,gain=512,hpf=300,lsb=.0000063,spike_threshold=5).items():
            d["settings/"+k]=[v]
        d["settings/mapping"]=np.array([tuple(r) for r in plan["mapping"]],dtype=[("channel","i4"),("electrode","i4")])
        raw=d.create_dataset("groups/routed/raw",(1024,n),dtype="u2",chunks=(1,2000),fillvalue=512)
        raw[stim,anchor:anchor+4]=811
        d["groups/routed/frame_nos"]=np.arange(n,dtype="u8")
        d["events"]=np.array([(anchor,2,1,raw_marker.encode())],dtype=[("frameno","u8"),("eventid","u4"),
            ("eventtype","u4"),("eventmessage","S90")])
        d["spikes"]=np.array([(k*20000,c) for k in (8,9,27) for c in (3,4)],dtype=[("frameno","u8"),("channel","u4")])

def scenario(root,label,failure=None):
    folder=root/(label+(failure or "ok"));folder.mkdir()
    mapping=[(i,i+1000) for i in range(7)]
    text="".join(f"{c}({e})" for c,e in mapping)
    route=folder/"recording.cfg";route.write_text(text)
    plan=dict(mapping=mapping,route=str(route),suggested={site:dict(channel=i,electrode=i+1000,unit=i+1)
            for i,site in enumerate("HWF")})
    params=s.configs(plan,folder)[label];row=plan["suggested"][label]
    b=dict(chip_id="fixture");mock=FakeMaxLab();mock.lsb="2.981901";base=mock.reply
    def reply(cmd):
        result=base(cmd)
        if "mea_array_get_config" in cmd:return text
        if "mea_array_query_stimulation_at_electrode" in cmd:return str(row["unit"]+(failure=="wrong_unit"))
        return result
    mock.reply=reply
    def report():
        return next((folder/label).glob("*/run.json"))
    def datafile():
        return Path(s.load(report())["h5_directory_on_server"])/"stimulation.raw.h5"
    def snapshot():
        return (4721,[str(datafile())] if "saving_start_recording" in mock.commands and "saving_stop_recording" not in mock.commands else [])
    clock=SimpleNamespace(now=100.)
    def sleep(dt):
        clock.now+=dt
        if "saving_start_recording" in mock.commands and "saving_stop_recording" not in mock.commands:
            h5=datafile()
            if not h5.exists() and failure!="missing_h5":h5.write_bytes(b"0"*70000)
            if dt==.75 and failure not in ("missing_h5","stale_h5"):
                with h5.open("ab") as f:f.write(b"frame")
            if failure=="missing_status" and "sequence_send test-sequence" not in mock.commands:
                path=report();obj=s.load(path);obj["status"]="starting";path.write_text(json.dumps(obj))
    fakeclock=SimpleNamespace(time=lambda:clock.now,monotonic=lambda:clock.now,sleep=sleep)
    original=s.postcheck
    def postcheck(path,b,plan,label,bits):
        marker=(f"amplitude {bits} event_id 2" if label=="W" else json.dumps(dict(amplitude=str(bits),event_id="2")))
        make_recording(path,plan,label,b["chip_id"],marker)
        return original(path,b,plan,label,bits)
    try:
        endpoint=s.transport.Endpoint("127.0.0.1",mock.server_address[1],2)
        assert endpoint.port!=7215
        with patch.object(s,"time",fakeclock),patch.object(s,"_server_snapshot",snapshot),patch.object(s,"stream_pid",return_value=91),patch.object(s,"postcheck",postcheck):
            try:
                s.run_point(label,params,folder,plan,b,endpoint,91,100,2500,SimpleNamespace(error=None,count=12))
                passed=True
            except RuntimeError:
                passed=False
        sent=mock.commands.count("sequence_send test-sequence")
        assert passed==(failure is None) and sent==int(failure is None),(failure,sent,passed)
        assert "saving_stop_recording" in mock.commands
        if passed:
            assert mock.commands.index("saving_start_recording")<mock.commands.index("sequence_send test-sequence")<mock.commands.index("saving_stop_recording")
            result=s.load(folder/(label+"-result.json"))
            assert result["onset_visible"] and result["stim_channel_recovered_at_12s"] and "late_state2" in result
        assert (folder/("attempt-"+label+".json")).exists()==passed
        return dict(site=label,scenario=failure or "normal",mock_sequence_send=sent,recorder_closed=True,state2_verified=passed)
    finally:mock.close()

def study_handoff(root):
    """Exercise the real binding, route loader, device constructor and RPC lifetime with hardware I/O replaced."""
    folder=root/"study";folder.mkdir()
    route=folder/"recording.cfg";route.write_text("".join(f"{i}({1000+i})" for i in range(64)))
    spots={k:dict(channel=i,electrode=1000+i,x_um=100.*i,y_um=50.*i) for i,k in enumerate("HWF")}
    plan=dict(suggested=spots,joint_units=dict(H=0,W=1,F=2),route=str(route),
              route_sha256=s.hashlib.sha256(route.read_bytes()).hexdigest(),region_count=3,
              distances_um={"H-W":112,"H-F":224,"W-F":112},minimum_distance_um=112,quadrant_midpoint_um=[100,50])
    for k in "HWF":s.write_once(folder/(k+"-result.json"),dict(onset_visible=True))
    b=dict(record_root=str(root),chip_id="fixture",organoid_id="fixture",session_id="fixture",replicate=0,
           device_flow="braindevice_5555_7215_v1",device_model="MaxOne",device_hostname=s.socket.gethostname(),
           acquisition_host="192.0.2.21",started_at=datetime.now(timezone.utc).isoformat(),threshold_h5_dataset=s.CFG.threshold_h5_dataset)
    def prepare(d):d.units=plan["joint_units"]
    def prime(d,*_):d.norm={"rehearsal":True};return []
    with patch.object(s,"localize",return_value=(folder,plan,{})),patch.object(s,"REG",root),patch.object(s,"Receiver"),patch.object(s,"_server_snapshot",return_value=(1,[])),patch.object(s.signal,"setitimer"),patch.object(s.Device,"preflight",return_value={"lsb_mv":2.981901}),patch.object(s.Device,"prepare_sites",prepare),patch.object(s.Device,"prime",prime):
        study=s.Study()
        result=study.open(b)
        assert study.dev.attempts==3 and result["binding"]==s.digest(result["resolved_binding"])
        assert result["resolved_binding"]["started_at"]==b["started_at"]
        assert s.load(folder/"paired"/"spatial-selection.json")["suggested"]==spots
        assert study.close()==study.close()==dict(closed=True,errors=[])
    return dict(original_start_preserved=True,prior_attempts=3,physical_output=False)

if __name__=="__main__":
    with tempfile.TemporaryDirectory() as tmp:
        rows=[scenario(Path(tmp),k) for k in "HWF"]
        rows.extend(scenario(Path(tmp),"H",failure) for failure in ("wrong_unit","missing_h5","stale_h5","missing_status"))
        print(json.dumps(dict(simulation=True,physical_output=False,results=rows,study_handoff=study_handoff(Path(tmp)))))
