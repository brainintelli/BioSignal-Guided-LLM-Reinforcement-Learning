"""Static report validation and an evidence-file digest manifest."""
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from PIL import Image

folder=Path(__file__).resolve().parent
class Parser(HTMLParser):
    def __init__(self):
        super().__init__();self.counts={};self.local_links=[];self.images=[]
    def handle_starttag(self,tag,attrs):
        self.counts[tag]=self.counts.get(tag,0)+1
        attrs=dict(attrs)
        if tag=='a':
            href=attrs.get('href','')
            if href and ':' not in href and not href.startswith('#'):
                self.local_links.append(href)
        if tag=='img':self.images.append(attrs['src'])
p=Parser();p.feed((folder/'report.html').read_text(encoding='utf-8'))
assert p.counts['h2']==10
assert p.counts['table']>=10
assert all((folder/x).is_file() for x in p.local_links)
assert len(p.images)==1 and p.images[0].startswith('data:image/png;base64,')
assert Image.open(folder/'rl_algorithm.png').size==(1640,1120)
ET.parse(folder/'rl_algorithm.svg')
assert '\ufffd' not in (folder/'report.md').read_text(encoding='utf-8')
verification=json.loads((folder/'verification.json').read_text())
assert verification['passed'] and verification['real_godot_replayed_steps']==288
post=json.loads((folder/'postcheck.json').read_text())
assert not post['remote_source_mismatches'] and not post['gpu_compute_apps_after']
result=dict(static_passed=True,html_sections=p.counts['h2'],html_tables=p.counts['table'],
            local_links_checked=len(p.local_links),diagram_xml_valid=True,diagram_png_dimensions=[1640,1120],
            browser_render_verified=False,browser_render_limit='Bundled browser absent; installed Edge failed to launch inside the filesystem sandbox. No browser screenshot or responsive-layout claim.',
            diagram_visually_inspected=True)
(folder/'render_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
manifest={x.name:{'bytes':x.stat().st_size,'sha256':hashlib.sha256(x.read_bytes()).hexdigest()}
          for x in sorted(folder.iterdir()) if x.is_file() and x.name!='artifact_manifest.json'}
(folder/'artifact_manifest.json').write_text(json.dumps(dict(created_utc=datetime.now(timezone.utc).isoformat(),files=manifest),indent=2),encoding='utf-8')
print(json.dumps(dict(**result,manifest_files=len(manifest))))
