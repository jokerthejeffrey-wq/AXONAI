"""50M-parameter AXON research checkpoint migration.

This is an *offline* versioned model experiment, not the live 3.4M Vercel
model. Expanded checkpoints require a new Blob-backed training protocol and
an inference benchmark before they can replace production.
"""
import argparse
from pathlib import Path
import torch

from settings import CONFIG, LARGE_CONFIG, TRAINING_PROFILES, parameter_count
from student import Student, load_weights, snapshot


def _copy_norm(dest, src):
    w = src.weight.shape[0]
    dest.weight[:w].copy_(src.weight)
    dest.bias[:w].copy_(src.bias)


@torch.no_grad()
def expand_legacy(source, *, seed=50224, target=None):
    """Initialize an expanded transformer, transferring matching legacy blocks.

    This is *not* guaranteed function preserving: wider layer norms and
    attention-head dimensions alter activations. A separate evaluation is
    required before making it the public chatbot.
    """
    if source.config != CONFIG:
        raise ValueError('Expected the 3.4M legacy architecture')
    if not isinstance(seed, int) or not 0 <= seed < 2**31:
        raise ValueError('Invalid initialization seed')
    if target is None:
        target = LARGE_CONFIG
    if (target['vocab_size'] != CONFIG['vocab_size'] or
        target['width'] < CONFIG['width'] or
        target['layers'] < CONFIG['layers'] or
        target['heads'] != CONFIG['heads'] or
        target['context'] < CONFIG['context'] or
        target['width'] % target['heads']):
        raise ValueError('Incompatible expansion target')

    # Preserve the caller's global RNG state.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        large = Student(dict(target))
    src_w = CONFIG['width']
    dst_w = target['width']
    large.token.weight[:, :src_w].copy_(source.token.weight)
    large.position.weight[:CONFIG['context'], :src_w].copy_(source.position.weight)
    _copy_norm(large.norm, source.norm)
    large.head.weight[:, :].zero_()
    large.head.weight[:, :src_w].copy_(source.head.weight)

    for src, dst in zip(source.blocks, large.blocks):
        _copy_norm(dst.n1, src.n1)
        _copy_norm(dst.n2, src.n2)
        for section in range(3):
            lo_src = section*src_w
            lo_dst = section*dst_w
            dst.qkv.weight[lo_dst:lo_dst+src_w, :src_w].copy_(
                src.qkv.weight[lo_src:lo_src+src_w])
            dst.qkv.bias[lo_dst:lo_dst+src_w].copy_(
                src.qkv.bias[lo_src:lo_src+src_w])
        dst.proj.weight[:src_w, :src_w].copy_(src.proj.weight)
        dst.proj.bias[:src_w].copy_(src.proj.bias)
        dst.ff[0].weight[:4*src_w, :src_w].copy_(src.ff[0].weight)
        dst.ff[0].bias[:4*src_w].copy_(src.ff[0].bias)
        dst.ff[2].weight[:src_w, :4*src_w].copy_(src.ff[2].weight)
        dst.ff[2].bias[:src_w].copy_(src.ff[2].bias)
    # Newly added blocks initially behave as residual identities.
    # Their internal weights remain trainable and learn during training.
    for block in large.blocks[CONFIG['layers']:]:
        block.proj.weight.zero_()
        block.proj.bias.zero_()
        block.ff[2].weight.zero_()
        block.ff[2].bias.zero_()
    return large


def create_checkpoint(legacy_path, output, seed=50224, profile="axon50m"):
    """Generate a complete, hashable, FLOAT32 safetensors research artifact."""
    from safetensors.torch import save_file
    old = Student()
    load_weights(old, Path(legacy_path))
    if profile not in ("axon11m", "axon50m"): raise ValueError("Unknown research profile")
    target = TRAINING_PROFILES[profile]
    upgraded = expand_legacy(old, seed=seed, target=target)
    count = sum(t.numel() for t in upgraded.parameters())
    if count != parameter_count(target):
        raise RuntimeError('Expanded architecture parameter mismatch')
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.partial')
    save_file(snapshot(upgraded), str(temporary))
    temporary.replace(output)
    return dict(path=str(output), parameter_count=count, bytes=output.stat().st_size)


def main(argv=None):
    p=argparse.ArgumentParser(description='Create an isolated 50M AXON checkpoint')
    p.add_argument('--from-legacy', type=Path, required=True,
                   help='existing verified small AXON .safetensors checkpoint')
    p.add_argument('--out', type=Path, default=Path.home()/'.axonai'/'50m'/'research.safetensors')
    p.add_argument('--profile', choices=['axon11m','axon50m'], default='axon11m')
    p.add_argument('--seed', type=int, default=50224)
    args=p.parse_args(argv)
    info=create_checkpoint(args.from_legacy,args.out,args.seed,args.profile)
    print(f"Created {info['parameter_count']:,}-parameter checkpoint "
          f"({info['bytes'] / 1e6:.1f} MB) at {info['path']}")
    print('Research only; this file is NOT the currently deployed chat model.')


if __name__=='__main__':
    main()
