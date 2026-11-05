# SPDX-License-Identifier: Apache-2.0

"""Inspect the platform CUDA API in a child process before narrowing visibility."""

from contextlib import redirect_stdout
import json
import sys
from uuid import UUID


def inspect_devices():
    import torch
    from ignition.preparation.host_memory import device_memory_info

    records, identities = [], set()
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        identity = UUID(bytes=bytes(properties.uuid.bytes))
        if not identity.int or identity in identities:
            raise ValueError('platform CUDA devices must expose distinct nonzero UUIDs')
        identities.add(identity)
        free, total = device_memory_info(index)
        if not 0 <= free <= total or total <= 0:
            raise ValueError(f'device {index} reported an invalid memory budget')
        records.append(dict(index=index, name=properties.name, uuid='GPU-' + str(identity),
            total_bytes=total, free_bytes=free, capability=(properties.major, properties.minor)))
    return records


if __name__ == '__main__':
    with redirect_stdout(sys.stderr):
        records = inspect_devices()
    print(json.dumps(records))
