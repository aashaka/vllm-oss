import enum
import os
import socket
import uuid
from platform import uname
from typing import Dict, List

import GPUtil
import psutil
import torch

from vllm._C import cuda_utils


class Device(enum.Enum):
    GPU = enum.auto()
    CPU = enum.auto()


class WorkerType(enum.Enum):
    PROMPT = enum.auto()
    TOKEN = enum.auto()
    MIXED = enum.auto()


class Counter:

    def __init__(self, start: int = 0) -> None:
        self.counter = start

    def __next__(self) -> int:
        i = self.counter
        self.counter += 1
        return i

    def reset(self) -> None:
        self.counter = 0


# Maximum number of sequences that can be in the system at a time
# Only used when sep_prompt_token is set in Parallel Config
# It determines the number of semaphores that will be used for KV cache transfer using MSCCL++
# This can be changed based on need
MAX_SLOT_IDS = 10


class SeqToSlotMapper:
    """ SeqToSlotMapper maps sequence ids to a limited set of slot ids.
    A slot is freed every time a sequence finishes. It is used to manage
    the semaphores for MSCCL++ proxy channels - there are as many semaphores
    as the number of slots. Each sequence is mapped to a different semaphore/slot,
    in order to allow fine-grained synchronization
    """
    def __init__(self):
        self.available_slotids = list(range(MAX_SLOT_IDS))
        self.seq_to_slot = {}

    def set_seq(self, seq_id):
        try:
            slot_id = self.available_slotids.pop(0)
        except IndexError:
            raise RuntimeError("No more slots available. Increase MAX_SLOT_IDS.")
        self.seq_to_slot[seq_id] = slot_id

    def free_seq(self, seq_id):
        slot_id = self.seq_to_slot.pop(seq_id)
        self.available_slotids.insert(0, slot_id)

    def get_slot_id(self, seq_id):
        return self.seq_to_slot[seq_id]


def is_hip() -> bool:
    return torch.version.hip is not None


def get_max_shared_memory_bytes(gpu: int = 0) -> int:
    """Returns the maximum shared memory per thread block in bytes."""
    # https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html
    cudaDevAttrMaxSharedMemoryPerBlockOptin = 97 if not is_hip() else 74
    max_shared_mem = cuda_utils.get_device_attribute(
        cudaDevAttrMaxSharedMemoryPerBlockOptin, gpu)
    return int(max_shared_mem)


def get_cpu_memory() -> int:
    """Returns the total CPU memory of the node in bytes."""
    return psutil.virtual_memory().total


def random_uuid() -> str:
    return str(uuid.uuid4().hex)


def in_wsl() -> bool:
    # Reference: https://github.com/microsoft/WSL/issues/4071
    return "microsoft" in " ".join(uname()).lower()


def get_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))  # Doesn't need to be reachable
    return s.getsockname()[0]


def get_open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def set_cuda_visible_devices(device_ids: List[int]) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, device_ids))


def get_total_num_gpus() -> int:
    return len(GPUtil.getGPUs())


def coalesce_blocks(block_list: List[int]):
    '''Coalesce of list of blocks to exploit contiguous chunks.
    '''
    if not block_list:
        return []
    sorted_block_list = sorted(block_list)
    ret = []
    current_block_start = sorted_block_list[0]
    current_block_length = 1
    for i in range(1, len(sorted_block_list)):
        if sorted_block_list[i] == sorted_block_list[i - 1] + 1:
            current_block_length += 1
        else:
            ret.append((current_block_start, current_block_length))
            current_block_start = sorted_block_list[i]
            current_block_length = 1
    ret.append((current_block_start, current_block_length))
    return ret


def coalesce_blocks_by_id(blocks_to_nw_dict: Dict[int, List[int]]):
    for cur_id in blocks_to_nw_dict:
        blocks_to_nw_dict[cur_id] = coalesce_blocks(blocks_to_nw_dict[cur_id])
    return blocks_to_nw_dict
