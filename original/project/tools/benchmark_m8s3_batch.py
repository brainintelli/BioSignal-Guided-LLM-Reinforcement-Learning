"""Run the existing equal-work protocol with explicitly bound batch policy."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent.m8s3_batch_contract import validate_config
from agent.m8s3_diag_contract import verify_registration_digest
from tools import benchmark_m8s3 as runner


def verify(reg):
    verify_registration_digest(reg)
    validate_config(reg['config'])
    for name, digest in reg['source_hashes'].items():
        if runner.sha(Path(reg['config']['project']) / name) != digest:
            raise ValueError('Source changed: ' + name)
    for item in reg['input_files'].values():
        if runner.sha(item['path']) != item['sha256']:
            raise ValueError('Input changed: ' + item['path'])


def make(c, mode):
    from agent.m8s3_batch_policy import CanonicalBatchPolicy
    from agent.m8s3_padded_policy import NativePaddedSemanticPolicy, PaddedSemanticPolicy
    baseline = NativePaddedSemanticPolicy if c.baseline_projection_layout == 'native' else PaddedSemanticPolicy
    cls = baseline if mode == 'baseline' else CanonicalBatchPolicy
    return cls(c.model, checkpoint=c.initial_checkpoint, parent_checkpoint=c.parent_checkpoint,
               seed=c.rng_seed, reset_critic=True, with_reference=True, max_tokens=c.max_tokens,
               prefix_layout=c.prefix_layout)


def main():
    # Reuse the unchanged timing, update counting and rollout protocol. This
    # binding is confined to this executable; the old runner remains unchanged.
    runner.make = make
    runner.verify = verify
    runner.main()


if __name__ == '__main__':
    main()
