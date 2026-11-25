# SPDX-License-Identifier: GPL-3.0-or-later
"""Opt-in two-rank Sonar integration for the experimental host pipeline."""
import ctypes
import logging
from importlib.metadata import version
from pathlib import Path
import socket
import threading
import uuid

import torch
import torch.distributed as dist
from sonar.distributed.device_communicators.pynccl import PyNcclCommunicator

logger = logging.getLogger(__name__)
_library = ctypes.CDLL(str(Path(__file__).with_name('libhoststaged.so')))
_library.hs_open.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
_library.hs_open.restype = ctypes.c_void_p
_library.hs_reduce.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
_library.hs_reduce.restype = ctypes.c_int
_library.hs_error.restype = ctypes.c_char_p
_library.hs_unlink.argtypes = [ctypes.c_char_p]
_library.hs_close.argtypes = [ctypes.c_void_p]
# CUDA graphs retain native pointers beyond communicator teardown. Process exit
# owns final CUDA/mmap cleanup, following the worker's lifetime.
_contexts = []
_original_init = PyNcclCommunicator.__init__
_original_reduce = PyNcclCommunicator.all_reduce


def initialize(self, *args, **kwargs):
    self._host_staged_handle = None
    self._host_staged_used = False
    self._host_staged_observed_sizes = set()
    self._host_staged_lock = threading.Lock()
    _original_init(self, *args, **kwargs)
    if not self.available or self.disabled or self.world_size != 2:
        return
    if not isinstance(self.group, dist.ProcessGroup):
        return
    metadata = (socket.gethostname(), str(torch.cuda.get_device_properties(self.device).uuid))
    peers = [None, None]
    dist.all_gather_object(peers, metadata, group=self.group)
    if peers[0][0] != peers[1][0] or peers[0][1] == peers[1][1]:
        return
    name = ('/sonar_hs_' + uuid.uuid4().hex).encode() if self.rank == 0 else None
    handle = None
    error = None
    if self.rank == 0:
        handle = _library.hs_open(name, 0, self.device.index, 1)
        if not handle:
            error = _library.hs_error().decode()
    announcement = [(name, error)]
    dist.broadcast_object_list(announcement, src=dist.get_process_group_ranks(self.group)[0], group=self.group)
    name, error = announcement[0]
    if error:
        raise RuntimeError('Host pipeline initialization failed: ' + error)
    if self.rank == 1:
        handle = _library.hs_open(name, 1, self.device.index, 0)
        if not handle:
            error = _library.hs_error().decode()
    errors = [None, None]
    dist.all_gather_object(errors, error, group=self.group)
    if self.rank == 0:
        _library.hs_unlink(name)
    if any(errors):
        if handle:
            _library.hs_close(handle)
        raise RuntimeError('Host pipeline initialization failed: ' + repr(errors))
    self._host_staged_handle = handle
    _contexts.append(handle)
    logger.warning('HOST_STAGED_READY rank=%s device=%s range=5..80MiB chunk=1MiB', self.rank, self.device)


def all_reduce(self, in_tensor, out_tensor=None, op=dist.ReduceOp.SUM, stream=None):
    handle = getattr(self, '_host_staged_handle', None)
    size = in_tensor.numel() * in_tensor.element_size()
    eligible = (handle is not None and not self.disabled and op == dist.ReduceOp.SUM
                and in_tensor.dtype == torch.bfloat16 and in_tensor.is_contiguous()
                and in_tensor.device == self.device and 5 * 2**20 <= size <= 80 * 2**20)
    if out_tensor is not None:
        eligible = (eligible and out_tensor.dtype == in_tensor.dtype
                    and out_tensor.device == in_tensor.device and out_tensor.is_contiguous()
                    and out_tensor.numel() == in_tensor.numel())
    if not eligible:
        return _original_reduce(self, in_tensor, out_tensor, op, stream)
    if not self._host_staged_used:
        logger.warning('HOST_STAGED_FIRST_USE rank=%s bytes=%s', self.rank, size)
        self._host_staged_used = True
    observed = self._host_staged_observed_sizes
    if size not in observed and len(observed) < 16:
        observed.add(size)
        logger.warning('HOST_STAGED_SIZE rank=%s bytes=%s', self.rank, size)
    if out_tensor is None:
        out_tensor = torch.empty_like(in_tensor)
    if stream is None:
        stream = torch.cuda.current_stream(self.device)
    with self._host_staged_lock:
        status = _library.hs_reduce(handle, in_tensor.data_ptr(), out_tensor.data_ptr(), size, stream.cuda_stream)
    if status:
        raise RuntimeError('Host pipeline enqueue failed: ' + _library.hs_error().decode())
    return out_tensor


def install():
    if version('sonar') != '0.26.0':
        raise RuntimeError('Revalidate the host pipeline before using a different Sonar version')
    PyNcclCommunicator.__init__ = initialize
    PyNcclCommunicator.all_reduce = all_reduce
