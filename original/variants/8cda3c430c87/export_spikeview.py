"""Read-only finalized H5 export; no acquisition, controller, stimulus, or GPU API."""
import argparse
import json
from pathlib import Path
from sr_data import sha,read,require,canonical


def export(source,output,qc_receipt):
    import h5py
    import numpy as np
    source=Path(source).resolve();output=Path(output).resolve();qc_receipt=Path(qc_receipt).resolve()
    require(source!=output and not output.exists(),'new output path required; original never overwritten')
    qc=read(qc_receipt)
    require(qc['schema']=='mea.28073.source_qc.v1' and qc['passed_for_declared_scope'] is True and qc['finalized'] is True,'passed finalized QC receipt required')
    parent=dict(sha256=sha(source),bytes=source.stat().st_size)
    require(qc['parent_raw']==parent,'QC receipt parent raw hash/size mismatch')
    source_stat=source.stat()
    with h5py.File(source,'r') as f:
        require(list(f['data_store'])==['data0000'],'one finalized store required')
        s=f['data_store/data0000'];r=s['groups/routed'];frames=r['frame_nos'][:]
        require(f['wellplate/id'][0]==b'28073','chip 28073 required')
        require(frames.ndim==1 and len(frames)>0 and np.all(frames[1:]==frames[:-1]+1),'contiguous integer frame domain required')
        require(r['frame_nos'].dtype.kind in 'iu' and list(r['raw'].shape)==[len(r['channels']),len(frames)],'parent raw shape invalid')
        provenance=dict(parent_raw=parent,qc_receipt_sha256=sha(qc_receipt),source_id=qc['source_id'],
                        contains_raw_waveforms=False,raw_interval=[int(frames[0]),int(frames[-1])+1],raw_shape=list(r['raw'].shape),
                        derivation='lossless_copy_of_recorded_spikes_events_settings_and_frame_ids; raw_waveforms_omitted',
                        scope='derived_spikeview_not_original_raw; waveform_QC_is_external_receipt')
        required=['wellplate/id','data_store/data0000/well_id','data_store/data0000/spikes',
                  'data_store/data0000/settings/mapping','data_store/data0000/settings/sampling','data_store/data0000/settings/lsb',
                  'data_store/data0000/groups/routed/frame_nos','data_store/data0000/groups/routed/channels']
        optional=['data_store/data0000/events','data_store/data0000/stimuli','data_store/data0000/settings/gain']
        with h5py.File(output,'x') as out:
            out.attrs['schema']='mea.spikeview.v1';out.attrs['provenance_json']=canonical(provenance)
            for name in required+[p for p in optional if p in f]:
                ds=f[name]
                if not isinstance(ds,h5py.Dataset):
                    f.copy(ds,out.require_group(str(Path(name).parent).replace('\\','/')),name=Path(name).name)
                    continue
                kwargs=dict(compression='gzip',compression_opts=4,shuffle=True) if ds.ndim>0 and ds.size and ds.dtype.kind!='O' else {}
                target=out.create_dataset(name,data=ds[()],dtype=ds.dtype,**kwargs)
                for key,val in ds.attrs.items():target.attrs[key]=val
            out.flush()
    require(source.stat().st_size==source_stat.st_size and source.stat().st_mtime_ns==source_stat.st_mtime_ns,'parent changed during export')
    require(sha(source)==parent['sha256'],'parent content changed during export')
    return dict(schema='mea.spikeview.export_receipt.v1',path=str(output),sha256=sha(output),bytes=output.stat().st_size,
                parent_raw=parent,qc_receipt_sha256=sha(qc_receipt),raw_interval=provenance['raw_interval'],contains_raw_waveforms=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('output');p.add_argument('--qc-receipt',required=True);args=p.parse_args()
    print(json.dumps(export(args.source,args.output,args.qc_receipt),indent=2))
