"""Process-group handling for single-process and torchrun (multi-GPU) execution."""
import datetime
import os

import numpy as np
import torch
import torch.distributed as dist

from common import get_device


class Context:
    """
    Rank/world information and collective helpers. Without torchrun (WORLD_SIZE unset) everything degrades to
    a single process, so the same code runs on a laptop CPU and on a multi-GPU node.
    """

    def __init__(self, device="auto", timeout_hours=12):
        self.world = int(os.environ.get("WORLD_SIZE", 1))
        self.rank = int(os.environ.get("RANK", 0))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.enabled = self.world > 1
        if not self.enabled:
            self.device = get_device(device)
            return
        use_cuda = torch.cuda.is_available() and device != "cpu"
        if use_cuda:
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device("cuda", self.local_rank)
        else:
            self.device = torch.device("cpu")
        dist.init_process_group("nccl" if use_cuda else "gloo", timeout=datetime.timedelta(hours=timeout_hours))

    @property
    def is_main(self):
        return self.rank == 0

    def barrier(self):
        if self.enabled:
            dist.barrier()

    def shard(self, items):
        """This rank's share of a list/array (round robin)."""
        return items[self.rank::self.world]

    def gather_dicts(self, d):
        """Merge a dict from every rank into one dict on every rank."""
        if not self.enabled:
            return d
        parts = [None] * self.world
        dist.all_gather_object(parts, d)
        out = {}
        for p in parts:
            out.update(p)
        return out

    def broadcast_object(self, obj):
        if not self.enabled:
            return obj
        box = [obj]
        dist.broadcast_object_list(box, src=0)
        return box[0]

    def close(self):
        if self.enabled:
            dist.destroy_process_group()


def gather_rows(array, ids):
    """Rows `ids` of a (memory-mapped) array as float32 numpy, reading in sorted order."""
    ids = np.asarray(ids, dtype=np.int64)
    out = np.empty((len(ids), array.shape[1]), dtype=np.float32)
    if len(ids):
        order = np.argsort(ids)
        out[order] = array[ids[order]]
    return out
