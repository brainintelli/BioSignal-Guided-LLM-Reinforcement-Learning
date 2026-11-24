"""Actual CUDA checks for an isolated optional kernel, not a training trial.

The independent reference and tolerances follow upstream v1.7.0 tests:
https://github.com/Dao-AILab/causal-conv1d/blob/v1.7.0/tests/test_causal_conv1d.py
This file tests the real Qwen convolution shape; it never saves model weights.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve previous probe evidence')
    occupied = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name',
                                       '--format=csv,noheader'], text=True).strip()
    if occupied:
        raise RuntimeError('Exclusive GPU required; left processes untouched: ' + occupied)
    import torch
    import causal_conv1d
    from causal_conv1d.causal_conv1d_interface import causal_conv1d_fn, causal_conv1d_ref
    torch.set_num_threads(8)
    rows = []
    for channel_last in (False, True):
        for weight_dtype in (torch.float32, torch.bfloat16):
            torch.manual_seed(98401)
            x = torch.randn((4, 1536, 6144) if channel_last else (4, 6144, 1536),
                            device='cuda', dtype=torch.bfloat16)
            if channel_last:
                x = x.transpose(1, 2)
            x.requires_grad_()
            weight = torch.randn(6144, 4, device='cuda', dtype=weight_dtype, requires_grad=True)
            xr = x.detach().clone().requires_grad_()
            # Use FP32 reference arithmetic for both storage formats. A BF16
            # conv followed by BF16 SiLU rounds the intermediate twice; that
            # is not a high-precision reference for a fused convolution.
            # Input weight values are still exactly the quantized test values.
            wr = weight.detach().float().clone().requires_grad_()
            out = causal_conv1d_fn(x, weight, activation='silu')
            reference = causal_conv1d_ref(xr, wr, activation='silu')
            torch.testing.assert_close(out, reference, rtol=1e-2, atol=5e-2)
            grad = torch.randn_like(out)
            out.backward(grad)
            reference.backward(grad)
            torch.testing.assert_close(x.grad, xr.grad, rtol=1e-2, atol=5e-2)
            wrtol, watol = (1e-3, 1e-3) if weight_dtype == torch.float32 else (1e-2, 5e-2)
            expected_weight_grad = wr.grad.to(weight_dtype)
            torch.testing.assert_close(weight.grad, expected_weight_grad, rtol=wrtol, atol=watol)
            difference = lambda a, b: float((a.float() - b.float()).abs().max())
            errors = dict(output_max_abs=difference(out, reference),
                          input_gradient_max_abs=difference(x.grad, xr.grad),
                          weight_gradient_max_abs=difference(weight.grad, expected_weight_grad))
            timing = {}
            for label, function in [('optimized', causal_conv1d_fn), ('reference', causal_conv1d_ref)]:
                samples = []
                for i in range(12):
                    x.grad = weight.grad = None
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    function(x, weight, activation='silu').backward(grad)
                    torch.cuda.synchronize()
                    if i >= 2:
                        samples.append(time.perf_counter() - start)
                timing[label] = dict(median_seconds=statistics.median(samples), samples_seconds=samples)
            row = dict(channel_last=channel_last, weight_dtype=str(weight_dtype), shape=list(x.shape),
                       reference_weight_arithmetic='float32_from_exact_test_values', **errors,
                       forward_and_input_gradient_tolerance=dict(rtol=1e-2, atol=5e-2),
                       weight_gradient_tolerance=dict(rtol=wrtol, atol=watol), timing=timing)
            rows.append(row)
            print(json.dumps(dict(event='kernel_case_passed', **row)), flush=True)
            del x, weight, xr, wr, out, reference, grad
            torch.cuda.empty_cache()
    result = dict(passed=True, scope='isolated_kernel_forward_backward_and_timing', cases=rows,
                  torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
                  kernel_version=causal_conv1d.__version__, kernel_path=causal_conv1d.__file__,
                  probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  model_loaded=False, new_environment_decisions=0, checkpoint_saved=False,
                  survival_evaluated=False, release_authorized=False)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
