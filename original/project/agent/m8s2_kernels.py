"""Pin FLA autotune choices so a policy is independent of prior process traffic."""
import gc

def pin_kernels():
    import fla.ops.gated_delta_rule.chunk
    from triton.runtime.autotuner import Autotuner
    chosen={}
    for obj in gc.get_objects():
        if type(obj) is not Autotuner or 'fla.' not in str(obj.fn):continue
        name=str(obj.fn)
        preferred={'num_warps':4,'num_stages':3,'BT':32,'BV':32,'BK':64}
        if 'chunk_fwd_kernel_o)' in name:preferred.update(BK=128,BV=128,num_warps=8)
        if 'chunk_scaled_dot_kkt_fwd_kernel)' in name:preferred.update(num_warps=8,num_stages=2)
        if 'recompute_w_u_fwd_kernel)' in name:preferred.update(num_warps=2)
        if 'l2norm_fwd_kernel)' in name:preferred.update(num_stages=2)
        # Ada has 99 KiB usable shared memory. The dhu backward pipeline at
        # three stages needs 132 KiB for K=V=128; use its two-stage variant.
        if 'chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64)' in name:preferred.update(num_stages=2)
        def rank(c):
            options={**c.kwargs,'num_warps':c.num_warps,'num_stages':c.num_stages}
            return (sum(abs(float(v)-preferred[k]) for k,v in options.items() if k in preferred),str(c))
        config=min(obj.configs,key=rank)
        obj.configs=[config];obj.cache.clear()
        chosen[name]=str(config)
    if not chosen:raise RuntimeError('Expected FLA autotuners were not loaded')
    return dict(sorted(chosen.items()))
