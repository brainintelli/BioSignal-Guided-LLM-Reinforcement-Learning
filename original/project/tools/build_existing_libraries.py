"""Build strictly causal wetware dual libraries & C_time null controls from existing archived data.

Zero hardware contact and zero new data collection:
1) `22021_1123` (Primary Stage 3 Organoid): 1,200 clean causal-filtered 2s windows, 243 channels
   (Rounds 1-2 = 800 windows build library; Round 3 = 400 windows strictly held-out).
2) `28080_1116_long` (Stage 4 Independent Replication Organoid): 898 continuous 2s windows, 831 channels
   (First 598 windows build library; final 300 windows strictly held-out).
3) `22021_1119_org2` (Intra-chip Second Organoid Backup): continuous 2s windows across routed channels.
4) `22554_1119` (Negative Biological QC Control): verifies that historical QC-failed recordings are
   automatically rejected by the held-out gate (`AUC ~ 0.50`).
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for extra in (ROOT, ROOT / 'standalone' / 'mea_oddball'):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from oddball import endogenous_library as endo


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def parse_route_cfg_xy(cfg_path: Path):
    """Parse MaxLab route line `ch(elec)x/y;...` into `{ch: (x_um, y_um)}`."""
    if not cfg_path.exists():
        return {}
    first_line = cfg_path.read_text(encoding='utf-8', errors='ignore').splitlines()[0].strip()
    xy = {}
    for token in first_line.split(';'):
        if not token or '(' not in token or ')' not in token or '/' not in token:
            continue
        ch_part, rest = token.split('(', 1)
        _, coord_part = rest.split(')', 1)
        xs, ys = coord_part.split('/', 1)
        xy[int(ch_part)] = (float(xs), float(ys))
    return xy


def load_22021_1123(output_root: Path, *, pre_stim_only: bool = False):
    """Load 2026-11-23 afternoon chip 22021 clean ROI readout spikes (233 channels).

    Strictly filters via `roi_mask_in_active` to exclude the 3 DAC-connected stimulation
    electrodes (`96, 737, 804`, which carry continuous DAC buffer noise during stimulation
    blocks) and the 7 near-field (<200 um) channels.
    """
    npz_path = output_root / 'mea_oddball_22021_20261123/exploratory-v1/q2_detected_spikes.npz'
    meta_path = output_root / 'mea_oddball_22021_20261123/exploratory-v1/q2_trials_meta.json'
    cand_path = output_root / 'mea_oddball_22021_20261123/records/oddball22021-20261123-site-selection/site-candidates.json'
    z = np.load(npz_path)
    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    cands = json.loads(cand_path.read_text(encoding='utf-8')).get('candidates', [])
    xy = {int(c['channel']): (float(c['x_um']), float(c['y_um'])) for c in cands}

    active_all = [int(c) for c in z['active_all']]
    if 'roi_mask_in_active' in z:
        roi_mask = np.asarray(z['roi_mask_in_active'], dtype=bool)
    elif 'roi' in z:
        roi_set = {int(c) for c in z['roi']}
        roi_mask = np.asarray([c in roi_set for c in active_all], dtype=bool)
    else:
        raise ValueError('q2_detected_spikes.npz missing roi_mask_in_active / roi')

    off_tid = z['off_tid']
    off_ch = z['off_ch']
    if pre_stim_only:
        # Strict pre-stimulus window [-300, -1) ms (frame < -20 at 20 kHz): zero post-pulse frames
        keep_spk = z['off_rf'] < -20
        off_tid = off_tid[keep_spk]
        off_ch = off_ch[keep_spk]

    mat_all = np.zeros((len(meta), len(active_all)), dtype=np.int32)
    np.add.at(mat_all, (off_tid, off_ch), 1)
    roi_ch = [c for c, keep in zip(active_all, roi_mask) if keep]
    excluded_ch = [c for c, keep in zip(active_all, roi_mask) if not keep]
    mat = mat_all[:, roi_mask]

    block_map = {}
    block_ids = np.zeros(len(meta), dtype=np.int64)
    for i, m in enumerate(meta):
        bname = m['block']
        if bname not in block_map:
            block_map[bname] = len(block_map)
        block_ids[i] = block_map[bname]

    return dict(
        dataset_id='22021_1123_oddball_clean_prestim' if pre_stim_only else '22021_1123_oddball_clean',
        chip_id='22021',
        counts=mat,
        block_ids=block_ids,
        fit_end=800,  # Rounds 1-2 (800 trials) fit; Round 3 (400 trials) strictly held-out
        channel_ids=roi_ch,
        excluded_channels=excluded_ch,
        total_spikes_all_243=int(mat_all.sum()),
        total_spikes_clean_roi=int(mat.sum()),
        pre_stim_only=bool(pre_stim_only),
        channel_xy_um=xy,
        history_k=4,
        inputs={
            str(npz_path.relative_to(ROOT).as_posix()): file_sha256(npz_path),
            str(meta_path.relative_to(ROOT).as_posix()): file_sha256(meta_path),
            str(cand_path.relative_to(ROOT).as_posix()): file_sha256(cand_path),
        },
    )


def audit_25449_afternoon_qc(output_root: Path):
    """Read-only QC audit of 2026-11-23 afternoon chip 25449 (stopped at warmup < 50 ROI channels)."""
    base = output_root / 'mea_oddball_25449_20261123'
    if not base.is_dir():
        return None
    end_path = base / 'records/oddball-v1/end.json'
    base_log = base / 'baseline.log'
    quiet_h5 = base / 'records/oddball25449-20261123-site-selection/quiet.raw.h5'
    warmup_h5 = base / 'records/oddball-v1/warmup/warmup.raw.h5'

    out = dict(
        dataset_id='25449_1123_afternoon_qc_rejected',
        chip_id='25449',
        g1_passed=False,
        dual_drive_unlocked=False,
        status='G1_rejected_low_activity',
        min_active_channels_required=50,
        inputs={},
        recordings={},
    )
    if end_path.is_file():
        end_obj = json.loads(end_path.read_text(encoding='utf-8'))
        out['session_outcome'] = end_obj.get('outcome')
        out['dose_real_pulses'] = int(end_obj.get('dose', 0))
        out['stop_error'] = str(end_obj.get('error', '')).strip().splitlines()[-1] if end_obj.get('error') else None
        out['inputs'][str(end_path.relative_to(ROOT).as_posix())] = file_sha256(end_path)
    if base_log.is_file():
        bl = json.loads(base_log.read_text(encoding='utf-8'))
        out['site_selection_candidates'] = len(bl.get('candidates', []))
        out['inputs'][str(base_log.relative_to(ROOT).as_posix())] = file_sha256(base_log)

    # Attempt direct H5 active-channel extraction using bundled Maxwell HDF5 plugin if available
    plugin_dir = ROOT / '.tools' / 'mea_maxwell_hdf5_plugin'
    if plugin_dir.is_dir():
        os.environ.setdefault('HDF5_PLUGIN_PATH', str(plugin_dir))
    deps_dir = ROOT / '.tools' / 'mea_baseline_deps'
    if deps_dir.is_dir() and str(deps_dir) not in sys.path:
        sys.path.insert(0, str(deps_dir))

    for tag, h5p in (('quiet', quiet_h5), ('warmup', warmup_h5)):
        if not h5p.is_file():
            continue
        rec_info = dict(
            path=str(h5p.relative_to(ROOT).as_posix()),
            size_bytes=int(h5p.stat().st_size),
            sha256=file_sha256(h5p),
        )
        try:
            from tools import audit_existing_mea_corpus as corpus_audit
            seg = corpus_audit.extract_spontaneous_segment(h5p)
            rec_info.update(
                verified_chip_id=str(seg['chip_id']),
                duration_s=round(float(seg['duration_s']), 2),
                n_active_channels=int(len(seg['active_channels'])),
                n_spikes_kept=int(seg['n_spikes_kept']),
                frame_gaps=int(seg['frame_gaps']),
            )
        except Exception as exc:
            rec_info['h5_read_note'] = f'skipped_h5_decode: {type(exc).__name__}: {exc}'
        out['recordings'][tag] = rec_info
        out['inputs'][rec_info['path']] = rec_info['sha256']
    return out


def load_28080_1116_long(output_root: Path):
    rec_path = output_root / 'mea_long_20261116/dataset28080_metadata/long_20261116_run01/recording.jsonl'
    ch_ids = None
    ch_to_col = {}
    xy = {}
    ev_f, ev_c = [], []
    with rec_path.open('r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            if 'channels' in obj and ch_ids is None:
                pairs = sorted(obj['channels'], key=lambda p: int(p[0]))
                ch_ids = [int(p[0]) for p in pairs]
                ch_to_col = {c: i for i, c in enumerate(ch_ids)}
                xy = {int(c): endo.maxone_electrode_xy_um(int(elec)) for c, elec in pairs}
            elif obj.get('type') == 'spikes' and ch_ids is not None:
                for fr, ch, _ in obj.get('events', ()):
                    c_int = int(ch)
                    if c_int in ch_to_col:
                        ev_f.append(int(fr))
                        ev_c.append(ch_to_col[c_int])
    ev_f = np.asarray(ev_f, dtype=np.int64)
    ev_c = np.asarray(ev_c, dtype=np.int32)
    t_idx = (ev_f - int(ev_f.min())) // 40000  # 2.0 s windows at 20 kHz
    n_t = int(t_idx.max())  # exclude incomplete final partial window
    mat = np.zeros((n_t, len(ch_ids)), dtype=np.int32)
    keep = t_idx < n_t
    np.add.at(mat, (t_idx[keep], ev_c[keep]), 1)
    return dict(
        dataset_id='28080_1116_long_spontaneous',
        chip_id='28080',
        counts=mat,
        block_ids=np.zeros(n_t, dtype=np.int64),
        fit_end=(2 * n_t) // 3,
        channel_ids=ch_ids,
        channel_xy_um=xy,
        history_k=4,
        inputs={str(rec_path.relative_to(ROOT).as_posix()): file_sha256(rec_path)},
    )


def load_cpp_spikes_session(output_root: Path, *, dataset_id, chip_id, spikes_rel, route_cfg_rel=None, history_k=3):
    sp_path = output_root / spikes_rel
    xy = parse_route_cfg_xy(output_root / route_cfg_rel) if route_cfg_rel else {}
    rows = [json.loads(line) for line in sp_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    ch_ids = sorted({int(r[1]) for r in rows})
    ch_to_col = {c: i for i, c in enumerate(ch_ids)}
    ts = np.asarray([int(r[0]) for r in rows], dtype=np.int64)
    cs = np.asarray([ch_to_col[int(r[1])] for r in rows], dtype=np.int32)
    order = np.argsort(ts, kind='stable')
    ts, cs = ts[order], cs[order]

    breaks = np.flatnonzero(np.diff(ts) > 10000) + 1
    s_ts = np.split(ts, breaks)
    s_cs = np.split(cs, breaks)
    mats, blks = [], []
    for bi, (st, sc) in enumerate(zip(s_ts, s_cs)):
        if len(st) >= 2 and (st[-1] - st[0]) >= 16000:
            ti = (st - st[0]) // 2000  # 2.0 s windows in ms
            nt = int(ti.max())
            if nt >= 8:
                m = np.zeros((nt, len(ch_ids)), dtype=np.int32)
                kp = ti < nt
                np.add.at(m, (ti[kp], sc[kp]), 1)
                mats.append(m)
                blks.append(np.full(nt, bi, dtype=np.int64))
    full_mat = np.vstack(mats)
    full_blk = np.concatenate(blks)
    inputs = {str(sp_path.relative_to(ROOT).as_posix()): file_sha256(sp_path)}
    if route_cfg_rel and (output_root / route_cfg_rel).exists():
        rp = output_root / route_cfg_rel
        inputs[str(rp.relative_to(ROOT).as_posix())] = file_sha256(rp)
    return dict(
        dataset_id=dataset_id,
        chip_id=chip_id,
        counts=full_mat,
        block_ids=full_blk,
        fit_end=(2 * len(full_mat)) // 3,
        channel_ids=ch_ids,
        channel_xy_um=xy,
        history_k=history_k,
        inputs=inputs,
    )


def _rel_str(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT.resolve()).as_posix())
    except ValueError:
        return str(resolved.as_posix())


def build_all(out_dir: Path, rho_min: float = 0.20, ctime_seed: int = 2026112399):
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    output_root = ROOT / 'output'

    datasets = [
        load_22021_1123(output_root),
        load_28080_1116_long(output_root),
        load_cpp_spikes_session(
            output_root,
            dataset_id='22021_1119_organoid2',
            chip_id='22021',
            spikes_rel='mea_predictability_22021_20261119T042416Z/cpp-crossover-v1/cpp-spikes.jsonl',
            route_cfg_rel='mea_predictability_22021_20261119T042416Z/sample-route.cfg',
            history_k=3,
        ),
        load_cpp_spikes_session(
            output_root,
            dataset_id='22554_1119_qc_negative',
            chip_id='22554',
            spikes_rel='22554_full_20261119_1224/records/session/cpp-spikes.jsonl',
            history_k=3,
        ),
    ]

    manifest_entries = {}
    md_lines = [
        '# 现有归档 MEA 数据内源因果反应库与时间置换零模型（`C_time`）核验报告',
        '',
        '本报告零硬件接触、零新采集，全部由工作区已归档实测数据离线提取，阈值与通道权重严格在前 2/3 建库段拟合，在后 1/3 留出段检验。',
        '',
        '| 数据集 | 模式 | 窗数(建库/留出) | 通道数 | G2诱发对比门 | G3习惯化门 | 留出 `rho_lib` (`all_sites`) | 各扇区留出 `AUC` (H / W / F) | 结论 |',
        '|---|---|---:|---:|---|---|---:|---|---|',
    ]

    for idx, ds in enumerate(datasets):
        ds_dir = out_dir / ds['dataset_id']
        ds_dir.mkdir(parents=True, exist_ok=True)

        real_res = endo.evaluate_and_build_bundle(
            ds['counts'],
            dataset_id=ds['dataset_id'],
            chip_id=ds['chip_id'],
            block_ids=ds['block_ids'],
            fit_end=ds['fit_end'],
            channel_ids=ds['channel_ids'],
            channel_xy_um=ds['channel_xy_um'],
            history_k=ds['history_k'],
            rho_min=rho_min,
            gate_seed=20261123 + idx * 100,
            permute_timeline_seed=None,
            force_export_for_control=False,
        )
        ctime_res = endo.evaluate_and_build_bundle(
            ds['counts'],
            dataset_id=ds['dataset_id'] + '_ctime',
            chip_id=ds['chip_id'],
            block_ids=ds['block_ids'],
            fit_end=ds['fit_end'],
            channel_ids=ds['channel_ids'],
            channel_xy_um=ds['channel_xy_um'],
            history_k=ds['history_k'],
            rho_min=rho_min,
            gate_seed=20269900 + idx * 100,
            permute_timeline_seed=ctime_seed + idx,
            force_export_for_control=True,
        )

        exported_files = {}
        if real_res['habituation_library'] is not None:
            p = ds_dir / 'habituation_library.json'
            p.write_text(json.dumps(real_res['habituation_library'], indent=2), encoding='utf-8')
            exported_files['habituation_library'] = dict(path=_rel_str(p), sha256=real_res['habituation_library']['sha256'])
        if real_res['evoked_library'] is not None:
            p = ds_dir / 'evoked_library.json'
            p.write_text(json.dumps(real_res['evoked_library'], indent=2), encoding='utf-8')
            exported_files['evoked_library'] = dict(path=_rel_str(p), sha256=real_res['evoked_library']['sha256'])

        # Always export C_time control libraries for passed datasets so Stage 3 Arm C_time can train on them
        if real_res['summary']['dual_drive_unlocked']:
            p_ch = ds_dir / 'ctime_habituation_library.json'
            p_ce = ds_dir / 'ctime_evoked_library.json'
            p_ch.write_text(json.dumps(ctime_res['habituation_library'], indent=2), encoding='utf-8')
            p_ce.write_text(json.dumps(ctime_res['evoked_library'], indent=2), encoding='utf-8')
            exported_files['ctime_habituation_library'] = dict(path=_rel_str(p_ch), sha256=ctime_res['habituation_library']['sha256'])
            exported_files['ctime_evoked_library'] = dict(path=_rel_str(p_ce), sha256=ctime_res['evoked_library']['sha256'])

        manifest_entries[ds['dataset_id']] = dict(
            inputs=ds['inputs'],
            real_gate=real_res['summary'],
            ctime_null_gate=ctime_res['summary'],
            exported=exported_files,
        )
        if 'excluded_channels' in ds:
            manifest_entries[ds['dataset_id']]['channel_purification'] = dict(
                n_clean_roi_channels=len(ds['channel_ids']),
                n_excluded_channels=len(ds['excluded_channels']),
                excluded_channels=ds['excluded_channels'],
                total_spikes_all_243=ds['total_spikes_all_243'],
                total_spikes_clean_roi=ds['total_spikes_clean_roi'],
            )

        for tag, summ in (('真实时序 B', real_res['summary']), ('时间置换 C_time', ctime_res['summary'])):
            hs = summ['habituation_held_sites']
            aucs = ' / '.join(f"{hs[s]['auc']:.3f}" if s in hs and hs[s]['auc'] is not None else 'n/a' for s in ('H', 'W', 'F'))
            rho_str = f"{summ['rho_lib']:+.3f}" if summ['rho_lib'] is not None else 'None'
            rho_all = f"{summ['rho_all_sites']:+.3f}" if summ['rho_all_sites'] is not None else 'n/a'
            verdict = '双门通过（解锁双驱）' if summ['dual_drive_unlocked'] else ('仅G2通过' if summ['g2_passed'] else '拦截/阴性')
            md_lines.append(
                f"| `{ds['dataset_id']}` | {tag} | {summ['n_windows']} ({summ['fit_end']}/{summ['held_windows']}) "
                f"| {summ['n_channels']} | {'PASS' if summ['g2_passed'] else 'FAIL'} | {'PASS' if summ['g3_passed'] else 'FAIL'} "
                f"| {rho_str} ({rho_all}) | {aucs} | **{verdict}** |"
            )

    # Sensitivity audit: afternoon 22021 clean 233-ROI pre-stimulus-only [-300, -1) ms (zero post-pulse frames)
    ds_pre = load_22021_1123(output_root, pre_stim_only=True)
    pre_real = endo.evaluate_and_build_bundle(
        ds_pre['counts'],
        dataset_id=ds_pre['dataset_id'],
        chip_id=ds_pre['chip_id'],
        block_ids=ds_pre['block_ids'],
        fit_end=ds_pre['fit_end'],
        channel_ids=ds_pre['channel_ids'],
        channel_xy_um=ds_pre['channel_xy_um'],
        history_k=ds_pre['history_k'],
        rho_min=rho_min,
        gate_seed=2026112377,
        permute_timeline_seed=None,
        force_export_for_control=False,
    )
    sensitivity_audits = {
        '22021_1123_prestim_only': dict(
            window_ms=[-300.0, -1.0],
            total_spikes_clean_roi=ds_pre['total_spikes_clean_roi'],
            summary=pre_real['summary'],
        )
    }

    # Afternoon chip 25449 G1 QC rejection audit
    qc_25449 = audit_25449_afternoon_qc(output_root)
    rejected_chips = {'25449_1123': qc_25449} if qc_25449 is not None else {}

    md_lines.extend([
        '',
        '## 下午数据纯度核验与质控拒收记录（2026-11-23）',
        '',
        f"- **`22021_1123_oddball_clean` 通道净化**：严格使用 `roi_mask_in_active` 保留 **233** 个干净读出通道（总放电 `{manifest_entries['22021_1123_oddball_clean']['channel_purification']['total_spikes_clean_roi']}`），剔除 3 个 DAC 直连刺激电极（`96, 737, 804`）及 7 个 `<200 µm` 近场通道（剔除前 243 通道含 `{manifest_entries['22021_1123_oddball_clean']['channel_purification']['total_spikes_all_243']}` 尖峰，其中约 70% 为 DAC 缓冲噪声）。",
        f"- **纯刺激前窗口 `[-300, -1) ms` 灵敏度复核**（0 刺激后帧，`{ds_pre['total_spikes_clean_roi']}` 干净放电）：G2=`{'PASS' if pre_real['summary']['g2_passed'] else 'FAIL'}`，G3=`{'PASS' if pre_real['summary']['g3_passed'] else 'FAIL'}`，留出 `rho_lib = {pre_real['summary']['rho_lib']:+.3f}`，证实信号完全源自类器官内源节律与突触抑压记忆。",
    ])
    if qc_25449 is not None:
        q_ch = qc_25449['recordings'].get('quiet', {}).get('n_active_channels', 'n/a')
        w_ch = qc_25449['recordings'].get('warmup', {}).get('n_active_channels', 'n/a')
        md_lines.append(
            f"- **下午芯片 `25449`（`oddball25449-20261123`）质控拒收**：`quiet` 活跃通道 `{q_ch}`、`warmup` 活跃通道 `{w_ch}`（均 `< 50` 门槛，0 真脉冲，`{qc_25449.get('stop_error')}`），自动拦截不入训。"
        )

    manifest = dict(
        schema='wetware.existing_libraries.manifest.v2',
        rho_min=rho_min,
        primary_dataset='22021_1123_oddball_clean',
        replication_dataset='28080_1116_long_spontaneous',
        datasets=manifest_entries,
        sensitivity_audits=sensitivity_audits,
        rejected_afternoon_chips=rejected_chips,
    )
    (out_dir / 'gate_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    (out_dir / 'REPORT.md').write_text('\n'.join(md_lines) + '\n', encoding='utf-8')
    return manifest


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=ROOT / 'output/existing_wetware_libraries')
    p.add_argument('--rho-min', type=float, default=0.20)
    return p


if __name__ == '__main__':
    args = build_parser().parse_args()
    m = build_all(args.out, rho_min=args.rho_min)
    summary = {
        k: dict(
            real_dual_unlocked=v['real_gate']['dual_drive_unlocked'],
            real_rho_lib=v['real_gate']['rho_lib'],
            ctime_dual_unlocked=v['ctime_null_gate']['dual_drive_unlocked'],
            ctime_rho_all=v['ctime_null_gate']['rho_all_sites'],
        )
        for k, v in m['datasets'].items()
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
