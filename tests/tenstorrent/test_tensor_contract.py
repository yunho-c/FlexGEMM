"""Hardware checks of the evaluator's residency gate."""

import torch

from benchmarks.tenstorrent.evaluate import tensor_contract
from test_explicit import device


def test_tensor_contract_rejects_host_l1_and_wrong_format(device):
    import ttnn
    values = torch.ones(32, 32, dtype=torch.bfloat16)
    host = ttnn.from_torch(values, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    dram = ttnn.to_device(host, device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    l1 = ttnn.to_memory_config(dram, ttnn.L1_MEMORY_CONFIG)
    wrong_layout = ttnn.to_layout(dram, ttnn.ROW_MAJOR_LAYOUT)
    wrong_dtype = ttnn.typecast(dram, ttnn.float32)
    assert tensor_contract(dram, device)["tensor_contract_valid"]
    assert not tensor_contract(host, device)["device_valid"]
    assert not tensor_contract(host, device)["tensor_contract_valid"]
    assert tensor_contract(l1, device)["device_valid"]
    assert not tensor_contract(l1, device)["dram_valid"]
    assert not tensor_contract(l1, device)["tensor_contract_valid"]
    assert not tensor_contract(wrong_layout, device)["tensor_contract_valid"]
    assert not tensor_contract(wrong_dtype, device)["tensor_contract_valid"]
