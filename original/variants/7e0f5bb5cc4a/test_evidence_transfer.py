import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from unpack_evidence import unpack
from export_evidence import export


class EvidenceTransferTests(unittest.TestCase):
    def bundle(self,root,name='experiment01/check.json',payload=b'{"value":1}',declared=None):
        archive=root/'evidence.zip';receipt=root/'receipt.json'
        manifest=dict(schema='mea.evidence.subset.v1',full_model_weights_downloaded=False,
                      remote_root='/test-fixture',included={name:dict(bytes=len(payload),
                      sha256=hashlib.sha256(payload if declared is None else declared).hexdigest())})
        with zipfile.ZipFile(archive,'x') as z:
            z.writestr(name,payload);z.writestr('__bundle__/manifest.json',json.dumps(manifest))
        receipt.write_text(json.dumps(dict(archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                           archive_bytes=archive.stat().st_size,included_files=1)))
        return archive,receipt

    def test_roundtrip_retains_exact_bytes_and_marks_weight_subset(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);archive,receipt=self.bundle(root);out=root/'out'
            unpack(archive,receipt,out)
            self.assertEqual((out/'experiment01/check.json').read_bytes(),b'{"value":1}')
            result=json.loads((out/'download_verification.json').read_text())
            self.assertTrue(result['passed']);self.assertFalse(result['full_model_weights_downloaded'])
            with self.assertRaises(FileExistsError):unpack(archive,receipt,out)

    def test_path_traversal_is_rejected_before_output_directory(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);archive,receipt=self.bundle(root,'../escaped.json');out=root/'out'
            with self.assertRaises(ValueError):unpack(archive,receipt,out)
            self.assertFalse(out.exists());self.assertFalse((root/'escaped.json').exists())

    def test_valid_archive_with_wrong_member_digest_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);archive,receipt=self.bundle(root,payload=b'bad',declared=b'good');out=root/'out'
            with self.assertRaises(ValueError):unpack(archive,receipt,out)
            self.assertFalse((out/'download_verification.json').exists())

    def test_export_rejects_partial_experiment_before_creating_archive(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'experiment01').mkdir()
            (root/'independent_audit.json').write_text('{"passed":false,"cells":0}')
            (root/'job.json').write_text('{"accepted_complete":false}')
            (root/'experiment01/parallel_status.json').write_text('{"passed":false,"accepted_cells":0}')
            out=root/'out.zip'
            with self.assertRaises(ValueError):export(root,out)
            self.assertFalse(out.exists())


if __name__=='__main__':unittest.main()
