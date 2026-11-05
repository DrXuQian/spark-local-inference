# SPDX-License-Identifier: Apache-2.0

"""Rebind prepared actlizeLA pointwise inputs without retaining host ABI objects in workers."""


class PreparedActlizePointwise:
    def __init__(self, library, operation, *, errors, **kwargs):
        from actlize_la.serving import PreparedPointwise

        self.owner = PreparedPointwise(library, operation, errors=errors, **kwargs)
        self.handle = self.owner.handle
        self.inputs, self.buffers = self.owner.inputs, self.owner.buffers
        self.positions = {name: i for i, name in enumerate(self.owner.read_names)}
        self.device_errors = (errors,)

    def __getstate__(self):
        return {name: value for name, value in vars(self).items() if name != 'owner'}

    def __call__(self, **operands):
        import torch

        inputs = list(self.inputs)
        for name, tensor in operands.items():
            inputs[self.positions[name]] = tensor
        torch.ops.actlize_la.serving_out(inputs, self.buffers, self.handle)
