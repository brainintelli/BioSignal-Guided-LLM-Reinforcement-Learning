"""Bind existing real views/QC/stimulation receipts; never create a QC decision."""
import argparse
import json
from pathlib import Path
from sr_data import canonical,derive,read,require,sha,write


def assemble(output,session_label,sources,partitions_path):
    import h5py
    output=Path(output).resolve();base=output.parent
    require(base.is_dir() and not output.exists(),'new manifest in existing input directory required')
    require(session_label.startswith('temporary_'),'explicit temporary identity required')
    def ref(path):
        path=Path(path).resolve();require(path.is_relative_to(base),'all input references must be inside the output manifest directory')
        return dict(path=path.relative_to(base).as_posix(),sha256=sha(path),bytes=path.stat().st_size)
    entries=[]
    for sid,view,qcp,stimp in sources:
        qc=read(qcp);stim=read(stimp)
        require(qc['source_id']==sid and stim['source_id']==sid,'source IDs in actual receipts differ')
        require(qc['passed_for_declared_scope'] is True and qc['finalized'] is True,'only existing passed/finalized QC decisions accepted')
        with h5py.File(view,'r') as f:
            require(f.attrs.get('schema')=='mea.spikeview.v1','explicit spikeview required')
            provenance=json.loads(f.attrs['provenance_json']);well=int(f['data_store/data0000/well_id'][0])
        entries.append(dict(source_id=sid,kind='spikeview',**ref(view),well_id=well,raw_interval=provenance['raw_interval'],
                            parent_raw=qc['parent_raw'],qc_status='passed_for_declared_scope',qc_receipt=ref(qcp),
                            qc_exclusions=qc['excluded_intervals'],declared_out_of_scope_events=qc['declared_out_of_scope_events'],
                            stimulation=dict(alignment=stim['alignment'],events=stim['events'],evidence=ref(stimp))))
    partitions=read(partitions_path)
    require(isinstance(partitions,list),'predeclared partition plan must be a JSON array')
    manifest=dict(schema='mea.28073.qc_manifest.v1',chip_id='28073',session_label=session_label,biological_specimen_id=None,
                  provenance='real_acquisition',finalized=True,qc_status='passed_for_declared_scope',source_association='prerecorded_action_independent',
                  frame_clock='session_continuous_hardware_frame',sources=entries,partitions=partitions,
                  partition_plan_sha256=sha(partitions_path))
    pending=output.with_name(output.name+'.pending')
    write(pending,manifest)
    data=derive(pending)  # Any failed provenance/QC/budget check prevents the final manifest.
    require(not output.exists(),'manifest appeared during validation; do not overwrite')
    pending.rename(output)
    return dict(manifest=str(output),manifest_sha256=sha(output),windows=len(data['windows']),excluded_windows=len(data['excluded_windows']),
                stimulation_events=data['source']['stimulation_events'],qc_decisions_created=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True);p.add_argument('--session-label',required=True)
    p.add_argument('--source',nargs=4,action='append',required=True,metavar=('ID','VIEW_H5','QC_JSON','STIM_JSON'))
    p.add_argument('--partitions',required=True);a=p.parse_args()
    print(canonical(assemble(a.output,a.session_label,a.source,a.partitions)))
