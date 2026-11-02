"""Prepare independent public LiDAR geometry with synthetic BF16 operands.

This is an operator validation corpus, not a BEVFusion model benchmark.
The point-cloud and voxel-parameter sources must be supplied explicitly.
"""

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
import yaml

from benchmarks.tenstorrent import common
from benchmarks.tenstorrent.common import neighbor_oracle, reference, sha256, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--voxel-config', type=Path, required=True)
    p.add_argument('--source-reference', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--threads', type=int, default=8)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    source_digest = sha256(args.source)
    if source_digest != 'e85ace25655fa2ca177194bd77902db4006499034d75034f1a2c760105cdfe74':
        raise ValueError('expected the pinned Autoware test_pcd sample')
    cfg = yaml.safe_load(args.voxel_config.read_text())['/**']['ros__parameters']
    with np.load(args.source, allow_pickle=False) as data:
        points = data['points']
        xyz = np.stack([points[k] for k in ('x', 'y', 'z')], axis=1)
    lower = np.array(cfg['point_cloud_range'][:3], dtype=np.float32)
    upper = np.array(cfg['point_cloud_range'][3:], dtype=np.float32)
    size = np.array(cfg['voxel_size'], dtype=np.float32)
    keep = np.isfinite(xyz).all(1) & ((xyz >= lower) & (xyz < upper)).all(1)
    grid = np.floor((xyz[keep] - lower) / size).astype(np.int32)
    shape = np.trunc((upper - lower) / size).astype(np.int32)
    grid = grid[((grid >= 0) & (grid < shape)).all(1)]
    unique = np.unique(grid, axis=0)
    coords = torch.from_numpy(np.column_stack([np.zeros(len(unique), dtype=np.int32), unique]))
    mapping = neighbor_oracle(coords)
    edges = int((mapping >= 0).sum())
    published = json.loads(args.source_reference.read_text())['meta']
    if len(coords) != published['voxels'] or edges != published['sparse']['pairs']['subm1']:
        raise ValueError('independent voxel/neighbor counts disagree with the published source')
    args.output.mkdir(parents=True, exist_ok=False)
    contract = Path(common.__file__).with_name('contract.json')
    (args.output / 'contract.json').write_bytes(contract.read_bytes())
    provenance = {
        'kind': 'public_lidar_geometry_with_synthetic_operands',
        'scene': 'Autoware ground-segmentation test.pcd, transformed to base_link',
        'source': str(args.source), 'source_sha256': source_digest,
        'source_url': 'https://huggingface.co/changh95/bevfusion-p150/blob/300095a42e61075a3d5c9916620344c1067200b8/code/tt_bevfusion/samples/test_pcd.npz',
        'attribution_url': 'https://huggingface.co/changh95/bevfusion-p150/blob/300095a42e61075a3d5c9916620344c1067200b8/code/tt_bevfusion/samples/README.md',
        'license': 'Apache-2.0',
        'voxel_config_sha256': sha256(args.voxel_config),
        'voxel_config_url': 'https://huggingface.co/AutowareFoundation/bevfusion/blob/e1bf164b909d3c6c2642012f9b5ee4d827753220/ml_package_bevfusion_lidar.param.yaml',
        'source_reference_sha256': sha256(args.source_reference),
        'point_cloud_range': cfg['point_cloud_range'], 'voxel_size': cfg['voxel_size'],
        'points_in': len(xyz), 'points_in_range': int(keep.sum()),
        'voxel_count': len(coords), 'neighbor_interactions': edges,
        'published_counts_match': True,
        'coordinate_order': 'lexicographic XYZ voxel order, one batch',
        'operand_scope': 'independent synthetic BF16 features, weights and biases; not captured BEVFusion activations',
        'feature_variant_1': 'x * 0.5 + 0.03125, rounded to BF16',
    }
    manifest = {'version': 1, 'status': 'preparing', 'torch': torch.__version__,
                'contract_sha256': sha256(contract), 'cases': [],
                'scope': 'Independent real LiDAR geometry; synthetic operands. No model or end-to-end performance claim.',
                'preparation_source_sha256': sha256(__file__)}
    for index, (ci, co) in enumerate(((16, 16), (32, 64), (64, 64), (128, 128), (256, 256), (17, 35))):
        start = time.perf_counter()
        seed = 20262102 + index
        gen = torch.Generator().manual_seed(seed)
        x = torch.randn(len(coords), ci, generator=gen).bfloat16()
        weight = (torch.randn(co, 27, ci, generator=gen) / math.sqrt(27 * ci)).bfloat16()
        bias = (torch.randn(co, generator=gen) * .1).bfloat16()
        variants = [x, (x.float() * .5 + .03125).bfloat16()]
        expected = [reference(v, weight, mapping, bias) for v in variants]
        name = f'autoware-test-pcd-{len(coords)}-{ci}x{co}'
        path = args.output / (name + '.pt')
        torch.save({'coords': coords, 'features': variants, 'weight': weight,
                    'bias': bias, 'mapping': mapping, 'expected': expected}, path)
        entry = {'name': name, 'split': 'external_geometry_validation', 'file': path.name,
                 'sha256': sha256(path), 'points': len(coords), 'ci': ci, 'co': co,
                 'volume': 27, 'valid_interactions': edges, 'occurrences': None,
                 'topology_oracle_rows': len(coords), 'provenance': dict(provenance, seed=seed),
                 'prepare_seconds': time.perf_counter() - start}
        manifest['cases'].append(entry)
        write_json(args.output / 'manifest.json', manifest)
        print(name, 'prepared', flush=True)
    manifest['status'] = 'ready'
    write_json(args.output / 'manifest.json', manifest)


if __name__ == '__main__':
    main()
