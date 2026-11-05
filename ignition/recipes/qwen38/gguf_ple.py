# SPDX-License-Identifier: Apache-2.0

"""Original interleaved IQ4_NL PLE rows, staged by the shared native reader."""

from dataclasses import dataclass
import json

import torch

from .ple import TABLE_ROWS


@dataclass(frozen=True)
class GgufPLELayout:
    caps: object
    geometry: object


@dataclass(frozen=True)
class GgufPLECaps:
    device: torch.device
    eos_token_id: int
    head_count: int = 16


@dataclass(frozen=True)
class GgufPLEGeometry:
    geometry: object
    prime_sizes: torch.Tensor
    table_offsets: torch.Tensor
    multipliers: torch.Tensor

    @classmethod
    def allocate(cls, geometry, device):
        return cls(geometry, *(torch.tensor(getattr(geometry, name), dtype=torch.int64, device=device)
            for name in ('prime_sizes', 'table_offsets', 'multipliers')))


def external_layout(config, metadata, device):
    from actlize_la.serving import ple_geometry

    if (config['ngram_size'], config['heads_per_ngram'], config['ple_embed_dim']) != (3, 8, 2560):
        raise ValueError('external GGUF PLE requires order three and eight 160-wide heads per order')
    eos = metadata['qwen4exp.ple.eos_token_id']
    configured = config['eos_token_id']
    if type(eos) is not int or not 0 <= eos < config['vocab_size'] or eos != (configured[0] if isinstance(configured, list) else configured):
        raise ValueError('GGUF PLE EOS must match the frontend and belong to its vocabulary')
    geometry = ple_geometry(vocab_size=config['vocab_size'],
        alignment=config['make_ngram_vocab_size_divisible_by'],
        prime_sizes=metadata['qwen4exp.ple.head_vocab_sizes'],
        table_offsets=metadata['qwen4exp.ple.head_offsets'],
        multipliers=metadata['qwen4exp.ple.layer_multipliers'])
    if geometry.padded_vocab_size != TABLE_ROWS:
        raise ValueError('GGUF PLE hash geometry does not cover the stored rows')
    return GgufPLELayout(GgufPLECaps(torch.device(device), eos), geometry)


@dataclass
class GgufPLETable:
    encoding = 'GGUF_IQ4_NL'

    config: dict
    layout: GgufPLELayout
    geometry: object
    weights: torch.Tensor
    hashes: torch.Tensor
    mailbox: object
    ready: object
    sources: tuple
    sequence: torch.Tensor | None = None
    timings: object = None

    @classmethod
    def load(cls, manifest, device, *, tokens, progress=None, host_allocator=None):
        from ignition.preparation.runtime_package import FileSource

        manifest.validate_files()
        metadata = manifest.tensors['per_layer_token_embd.weight']
        if metadata.shape != (TABLE_ROWS, 160) or metadata.encoding != 'IQ4_NL' or metadata.nbytes != TABLE_ROWS * 90:
            raise ValueError('Qwen GGUF PLE requires 320001536 original IQ4_NL rows of width 160')
        config = json.loads((manifest.metadata_path / 'config.json').read_text())['text_config']
        external = (getattr(manifest, 'actlize', None) is not None
                    and getattr(manifest, 'quactlize_helpers', None) is not None)
        if external:
            layout = external_layout(config, manifest.metadata, device)
            caps, geometry = layout.caps, layout.geometry
        else:
            from b12x.sequence import ple_hash

            caps = ple_hash.Caps(device=device, max_tokens=tokens, max_seqs=tokens,
                vocab_size=config['vocab_size'], eos_token_id=manifest.metadata['qwen4exp.ple.eos_token_id'],
                max_order=config['ngram_size'], heads_per_order=config['heads_per_ngram'],
                dense_layer_ordinal=0, base_table_size=config['ngram_vocab_size_base'],
                table_alignment=config['make_ngram_vocab_size_divisible_by'])
            geometry = ple_hash.compute_geometry(caps,
                prime_sizes=torch.tensor(manifest.metadata['qwen4exp.ple.head_vocab_sizes'], dtype=torch.int64),
                table_offsets=torch.tensor(manifest.metadata['qwen4exp.ple.head_offsets'], dtype=torch.int64),
                multipliers=torch.tensor(manifest.metadata['qwen4exp.ple.layer_multipliers'], dtype=torch.int64))
        if geometry.padded_vocab_size != TABLE_ROWS:
            raise ValueError('GGUF PLE hash geometry does not cover the stored rows')
        rows = tokens * caps.head_count
        shape = (rows + 1,)
        if host_allocator is None:
            from b12x.sequence._shared.disk_table import MappedHostAllocation

            mailbox = MappedHostAllocation(shape, torch.int64, torch.device(device))
        else:
            mailbox = host_allocator('ple_requests', shape, torch.int64)
        try:
            mailbox.host_view.zero_()
            ready = torch.cuda.Event(external=True)
            ready.record(torch.cuda.current_stream(device))
            geometry_buffers = (GgufPLEGeometry.allocate(geometry, device) if external
                                else ple_hash.allocate_geometry(geometry, device=device))
            return cls(config, GgufPLELayout(caps, geometry), geometry_buffers,
                torch.zeros((rows, 90), dtype=torch.uint8, device=device),
                torch.full((tokens, caps.head_count), -1, dtype=torch.int64, device=device), mailbox, ready,
                ((FileSource(manifest.snapshot / metadata.shard, metadata.offset, metadata.nbytes),),))
        except BaseException:
            mailbox.close()
            raise

    def buffers(self):
        return dict(ple_weights=self.weights, ple_hashes=self.hashes,
                    ple_requests=self.mailbox.device_view, **self.timings.buffers())

    def events(self):
        return {'ple_ready': self.ready.cuda_event}

    def export_metadata(self):
        geometry = self.layout.geometry
        return dict(pad_token_id=self.layout.caps.eos_token_id, compressed_vocabulary=self.config['vocab_size'],
            tables=[dict(layer=1, rows=geometry.padded_vocab_size, embedding_width=160, scale_block=32, encoding='GGUF_IQ4_NL',
                weight_buffer='ple_weights', hash_buffer='ple_hashes', ready_event='ple_ready', request_buffer='ple_requests',
                weights=[dict(path=str(s.path), offset=s.offset, nbytes=s.nbytes) for s in self.sources[0]],
                hash=dict(method='EOS_XOR_MODULO', orders=[2] * 8 + [3] * 8,
                    multipliers=list(geometry.multipliers), primes=list(geometry.prime_sizes),
                    offsets=list(geometry.table_offsets)))])

    def resident_catalog(self):
        return dict(vocabulary=self.config['vocab_size'], tables=[dict(
            planes=[dict(row_bytes=90, segments=[dict(path=str(s.path), offset=s.offset, nbytes=s.nbytes)
                                               for s in self.sources[0]])],
            heads=self.layout.caps.head_count, weights='ple_weights', scales='', hashes='ple_hashes',
            event='ple_ready', request='ple_requests')])

    def close(self):
        self.mailbox.close()
