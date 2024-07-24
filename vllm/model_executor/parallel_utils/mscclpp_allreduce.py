import torch
import cupy as cp

from mscclpp import ProxyService
from mscclpp_benchmark import MscclppAllReduce1, MscclppAllReduce2, MscclppAllReduce3
from vllm.model_executor.parallel_utils.communication_op import broadcast
from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_group, get_msccl_tensor_model_parallel_group, get_msccl_tensor_model_parallel_rank)

# import netifaces as ni
# import ipaddress
# from mpi4py import MPI
# import mscclpp.comm as mscclpp_comm

max_elements = 33554432

def type_to_str(dtype):
    if dtype == torch.float16:
        return "__half"
    elif dtype == torch.float32:
        return "float"
    elif dtype == torch.int32:
        return "int"
    else:
        raise RuntimeError(f"Unknown data type {dtype}")

def torch_to_cupy_type(dtype):
    if dtype == torch.float16:
        return cp.float16
    elif dtype == torch.float32:
        return cp.float32
    elif dtype == torch.int32:
        return cp.int32
    else:
        raise RuntimeError(f"Unknown data type {dtype}")

def bench_time(niter: int, func):
    # capture cuda graph for nites of the kernel launch
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        stream.begin_capture()
        for i in range(niter):
            func(stream)
        graph = stream.end_capture()

    # now run a warm up round
    graph.launch(stream)

    # now run the benchmark and measure time
    start = cp.cuda.Event()
    end = cp.cuda.Event()

    start.record(stream)
    graph.launch(stream)
    end.record(stream)
    end.synchronize()

    return cp.cuda.get_elapsed_time(start, end) / niter * 1000.0

def find_best_config(mscclpp_call, niter, *args):
    best_time = 10000000.0
    for config in mscclpp_call.auto_tune():
        cur_time = bench_time(niter, mscclpp_call, *args)
        if cur_time < best_time:
            best_time = cur_time
            best_config = config

    # for MPI test
    # best_config = MPI.COMM_WORLD.bcast(best_config, root=0)
    # if MPI.COMM_WORLD.rank == 0:
    #     print(best_config, end="", flush=True)

    # for actual run
    best_config_tensor = torch.tensor(best_config, dtype=torch.int32, device=torch.cuda.current_device())
    best_config_tensor = broadcast(best_config_tensor, src=0, group=get_tensor_model_parallel_group())
    best_config = best_config_tensor.cpu().tolist()

    return best_config, best_time

def find_best_algo_config(mscclpp_algos, niter):
    assert len(mscclpp_algos) > 0
    best_time = 10000000.0
    best_algo = None
    best_config = None
    best_algo_id = None
    for i, algo in enumerate(mscclpp_algos):
        if algo is None:
            continue
        config, cur_time = find_best_config(algo, niter)
        if cur_time < best_time:
            best_time = cur_time
            best_algo = algo
            best_config = config
            best_algo_id = i
    # for MPI test
    # if MPI.COMM_WORLD.rank == 0:
    #     print(best_algo, end="", flush=True)
    # for actual run
    # if get_msccl_tensor_model_parallel_rank() == 0:
    #     print(best_algo, end="", flush=True)
    return best_algo_id, best_algo, best_config

class MscclppAllReduce:
    # Buffers used as input and output for MSCCL++ allreduce
    all_reduce_buff = {torch.float16: None, torch.float32: None, torch.int: None}
    all_reduce_buff_out = {torch.float16: None, torch.float32: None, torch.int: None}

    # Cache for allreduce kernels
    ar_kernel = {}

    # Per-size best algo and config
    best_configs = {}

    BUILT_KERNELS = False

    @classmethod
    def build_kernels(cls, data_type): #, mscclpp_group):
        # Only build kernels once
        if cls.BUILT_KERNELS:
            return

        mscclpp_group = get_msccl_tensor_model_parallel_group()

        cls.all_reduce_buff[data_type] = torch.zeros(max_elements, dtype=data_type, device=torch.cuda.current_device())
        cls.all_reduce_buff_out[data_type] = torch.zeros(max_elements, dtype=data_type, device=torch.cuda.current_device())
        cp.cuda.runtime.deviceSynchronize()

        proxy_service = ProxyService()
        # Compile kernels for different allreduce algorithms using the same input buffer
        # These algorithms are for shards=8
        # AllReduce2 uses a lot more memory, and hence we limit the number of elements to 1M
        mscclpp_algos = [
            MscclppAllReduce1(mscclpp_group, cls.all_reduce_buff[data_type]),
            MscclppAllReduce2(mscclpp_group, cls.all_reduce_buff[data_type][:1048576], cls.all_reduce_buff_out[data_type][:1048576]),
            MscclppAllReduce3(mscclpp_group, cls.all_reduce_buff[data_type], proxy_service),
        ]
        cls.ar_kernel[data_type] = mscclpp_algos
        proxy_service.start_proxy()
        mscclpp_group.barrier()
        cls.BUILT_KERNELS = True


    @classmethod
    def get_best_ar_kernel(cls, nelem, data_type):
        assert nelem <= max_elements, f"AllReduce size is too large {nelem}"
        if (nelem, data_type) in cls.best_configs:
            # if the best config is already found for this data size, return the kernel
            algo_offset, config = cls.best_configs[(nelem, data_type)]
            mscclpp_call = cls.ar_kernel[data_type][algo_offset]
            mscclpp_call.memory = cls.all_reduce_buff[data_type][:nelem]
        else:
            mscclpp_algos = [algo for algo in cls.ar_kernel[data_type]]
            for algo in mscclpp_algos:
                algo.memory = cls.all_reduce_buff[data_type][:nelem]
            if nelem * algo.memory.element_size()  >= 2**20:
                mscclpp_algos[1] = None
            # Get best combination of algorithm and config
            algo_id, mscclpp_call, config = find_best_algo_config(mscclpp_algos, 20)
            # Cache the best config for this size (closest power of 2) and data type
            cls.best_configs[(nelem, data_type)] = (algo_id, config)

        mscclpp_call.set_params(*config)
        return mscclpp_call

    @classmethod
    def mscclpp_allreduce(cls, input_: torch.Tensor): #, mscclpp_group):
        nelem = input_.numel()
        cls.build_kernels(input_.dtype) #, mscclpp_group)
        mscclpp_ar_call = cls.get_best_ar_kernel(nelem, input_.dtype)
        # print(f"shape is : {input_.shape} nelem = {nelem} call.memory = {mscclpp_ar_call.memory.shape}", flush=True)
        mscclpp_ar_call.memory.copy_(input_.reshape(-1), non_blocking=True)
        if hasattr(mscclpp_ar_call, "memory_out"):
            mscclpp_ar_call.memory_out = cls.all_reduce_buff_out[input_.dtype][:nelem]
        return mscclpp_ar_call(torch.cuda.current_stream()).reshape(input_.shape)

# def is_valid(ip):
#     """
#     Check if the IP address is valid for connecting to other devices.
#     This excludes loopback (127.0.0.1) and link-local (169.254.x.x) addresses.
#     """
#     ip_obj = ipaddress.ip_address(ip)
#     return not (ip_obj.is_loopback or ip_obj.is_link_local or ip_obj.is_multicast)


# def get_netinterface_info():
#     """
#     Returns the name of the first network interface with a valid IP address that it finds.
#     """
#     interfaces = ni.interfaces()
#     for interface in interfaces:
#         addresses = ni.ifaddresses(interface)
#         if ni.AF_INET in addresses:
#             for addr in addresses[ni.AF_INET]:
#                 ip_address = addr["addr"]
#                 if is_valid(ip_address):
#                     print(f"Selected Interface: {interface}, IP Address: {ip_address}")
#                     return interface, ip_address
#     return None, None

# if __name__ == "__main__":
#     shm_comm = MPI.COMM_WORLD.Split_type(MPI.COMM_TYPE_SHARED, 0, MPI.INFO_NULL)
#     N_GPUS_PER_NODE = shm_comm.size
#     shm_comm.Free()
#     cp.cuda.Device(MPI.COMM_WORLD.rank % N_GPUS_PER_NODE).use()

#     # stream = torch.cuda.Stream()
#     # torch.cuda.set_stream(stream)

#     # create a MscclppGroup
#     network_interface, my_ip = get_netinterface_info()
#     root_ip = MPI.COMM_WORLD.bcast(my_ip, root=0)
#     ifIpPortTrio = network_interface + ":" + root_ip + ":50000"  # some random port
#     mscclpp_group = mscclpp_comm.CommGroup(
#         interfaceIpPortTrio=ifIpPortTrio, rank=MPI.COMM_WORLD.rank, size=MPI.COMM_WORLD.size
#     )

#     input_ = torch.ones(128,  device="cuda", dtype=torch.float16)
#     output = MscclppAllReduce.mscclpp_allreduce(input_, mscclpp_group)
#     torch.cuda.synchronize()
#     print(output)
#     assert torch.allclose(output, torch.ones(128, device="cuda", dtype=torch.float16) * MPI.COMM_WORLD.size)