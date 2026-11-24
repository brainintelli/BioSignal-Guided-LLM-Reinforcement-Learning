"""Verify archive bytes, paths and all selected files before using reports."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import zipfile


def unpack(archive_path,receipt_path,output):
    receipt=json.loads(receipt_path.read_text())
    with archive_path.open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
    if actual!=receipt['archive_sha256'] or archive_path.stat().st_size!=receipt['archive_bytes']:
        raise ValueError('Archive differs from server receipt')
    output=output.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        names=archive.namelist()
        if len(names)!=len(set(names)):raise ValueError('Duplicate archive path')
        index=json.loads(archive.read('__bundle__/manifest.json'))
        if index.get('schema')!='mea.evidence.subset.v1' or index.get('full_model_weights_downloaded') is not False:
            raise ValueError('Wrong bundle contract')
        files=index['included']
        if set(names)!=set(files)|{'__bundle__/manifest.json'} or len(files)!=receipt['included_files']:
            raise ValueError('Archive file set mismatch')
        destinations={}
        for name,metadata in files.items():
            parts=PurePosixPath(name)
            if parts.is_absolute() or '..' in parts.parts or '\\' in name or ':' in name:
                raise ValueError('Unsafe archive path')
            path=output.joinpath(*parts.parts).resolve()
            if not path.is_relative_to(output) or path==output:raise ValueError('Path escapes output')
            info=archive.getinfo(name)
            if stat.S_ISLNK(info.external_attr>>16) or info.file_size!=metadata['bytes']:
                raise ValueError('Unexpected file type/size')
            destinations[name]=path
        output.mkdir(parents=True,exist_ok=False)
        for name,path in destinations.items():
            path.parent.mkdir(parents=True,exist_ok=True)
            digest=hashlib.sha256()
            with archive.open(name) as source,path.open('xb') as target:
                for chunk in iter(lambda:source.read(1024*1024),b''):
                    digest.update(chunk);target.write(chunk)
            if digest.hexdigest()!=files[name]['sha256']:raise ValueError('File digest mismatch: '+name)
        (output/'bundle_manifest.json').write_text(json.dumps(index,indent=2),encoding='utf-8')
    result=dict(passed=True,verified_files=len(files),archive_sha256=actual,
                full_model_weights_downloaded=False,remote_root=index['remote_root'])
    (output/'download_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',required=True,type=Path);parser.add_argument('--receipt',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();unpack(args.archive,args.receipt,args.output)
