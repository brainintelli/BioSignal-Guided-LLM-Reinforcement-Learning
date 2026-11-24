from pathlib import Path
r=Path(__file__).resolve().parent
s=(r/'deploy_bounded_resume.py').read_text()
for old,new in [('bounded_code_v4','bounded_code_v5'),('bounded_v4','bounded_v5'),('bounded_resume_run','bounded_stable_run'),('bounded_resume_job','bounded_stable_job'),('bounded_resume_launch','bounded_stable_launch')]:s=s.replace(old,new)
(r/'deploy_bounded_stable.py').write_text(s)
s=(r/'poll_bounded.py').read_text()
s=s.replace('bounded_resume_job','bounded_stable_job').replace('multi_bounded_resume1','multi_bounded_resume2').replace('bounded_resume_run','bounded_stable_run')
(r/'poll_stable.py').write_text(s)
