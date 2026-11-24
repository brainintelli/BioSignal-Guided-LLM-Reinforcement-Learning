"""Command-line interface; dry-run never imports or connects the hardware API."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .config import describe, load_config


def main(argv=None):
    parser = argparse.ArgumentParser(description="MaxOne 独立刺激实验（JSON 参数）")
    parser.add_argument("config", type=Path, help="JSON params 或包含 params 的消息")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="仅验证参数和预览时序，不连接设备")
    mode.add_argument("--check", action="store_true", help="只读查询 MaxLab API 和 DAC 范围，不刺激")
    mode.add_argument("--run", action="store_true", help="运行真实刺激实验")
    parser.add_argument("--host", default="127.0.0.1", help="MaxLab API 地址")
    parser.add_argument("--port", type=int, default=7215, help="MaxLab API 端口（非 5555）")
    parser.add_argument("--timeout", type=float, default=60.0, help="单个 API 请求超时秒数")
    parser.add_argument("--output-dir", type=Path, default=Path("results"), help="本地结果根目录")
    parser.add_argument("--recording-dir", help="MaxLab 服务端 H5 根目录；远程连接时必填")
    args = parser.parse_args(argv)
    try:
        params = load_config(args.config)
        schedule = params.amplitude_schedule()
        if args.dry_run:
            print(json.dumps(describe(params, schedule), ensure_ascii=False, indent=2))
            return 0
        from .transport import Endpoint
        from .runner import check_connection, run_experiment
        endpoint = Endpoint(args.host, args.port, args.timeout)
        if args.check:
            print(json.dumps(check_connection(params, endpoint), ensure_ascii=False, indent=2))
            return 0
        return run_experiment(params, endpoint, args.output_dir,
                              recording_dir=args.recording_dir, schedule=schedule)
    except KeyboardInterrupt:
        print("已中断。", file=sys.stderr)
        return 130
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1
