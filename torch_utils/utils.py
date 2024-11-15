import datetime
import errno
import os
import time
from collections import defaultdict, deque

import torch
import torch.distributed as dist

from utils.logging import log


class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        """Create a SmoothedValue to track a series of values and provide access
        to smoothed values over a window or the global series average.

        Args:
            window_size (int): The size of the window. Defaults to 20.
            fmt (str): The format string to use when formatting the output.
                Defaults to "{median:.4f} ({global_avg:.4f})".
        """
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        """
        Update the SmoothedValue with a new value.

        Args:
            value (float): The new value to add to the series.
            n (int, optional): The number of times to add the value. Defaults to 1.

        This method appends the new value to the deque, increments the count by n,
        and updates the total sum with the new value multiplied by n.
        """
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Synchronizes the count and total across all processes.

        This method ensures that the `count` and `total` values are synchronized
        across all processes in a distributed setting. It does not, however,
        synchronize the internal deque of values.

        If the distributed environment is not available or initialized, the method
        returns immediately without performing any synchronization.

        The method uses `torch.distributed.all_reduce` to aggregate the `count` and
        `total` across all processes, ensuring that each process ends up with the
        same values for these attributes.

        **Warning: does not synchronize the deque!**
        """

        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        """
        The median value of the series of values in the deque.

        The median is calculated by converting the deque to a tensor and using
        the `median` method of the tensor. The result is returned as a float.

        This method is useful for getting a sense of the central tendency of
        the data in the deque, as it is less sensitive to outliers than the mean.

        Returns:
            float: The median value of the series of values in the deque.
        """
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        """
        The average value of the series of values in the deque.

        The average is calculated by converting the deque to a tensor and using
        the `mean` method of the tensor. The result is returned as a float.

        This method is useful for getting a sense of the central tendency of
        the data in the deque.

        Returns:
            float: The average value of the series of values in the deque.
        """
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        """
        The global average value of the series of values in the deque.

        The global average is calculated by summing all of the values in the deque
        and dividing by the count of values in the deque. The result is returned as
        a float.

        This method is useful for getting a sense of the central tendency of
        the data in the deque across all ranks/processes.

        Returns:
            float: The global average value of the series of values in the deque.
        """
        return self.total / self.count

    @property
    def max(self):
        """
        The maximum value of the series of values in the deque.

        The maximum is calculated by converting the deque to a list and using
        the built-in `max` method of the list. The result is returned as a float.

        This method is useful for getting a sense of the maximum of the data in the deque.

        Returns:
            float: The maximum value of the series of values in the deque.
        """
        return max(self.deque)

    @property
    def value(self):
        """
        The most recent value in the series of values in the deque.
        This property retrieves the last value that was added to the deque.

        Returns:
            The most recent value in the deque.
        """
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median, avg=self.avg, global_avg=self.global_avg, max=self.max, value=self.value
        )


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]
    data_list = [None] * world_size
    dist.all_gather_object(data_list, data)
    return data_list


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.inference_mode():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


class MetricLogger:
    """
    The MetricLogger class is a utility class for logging and tracking metrics during training or evaluation processes.
    It provides methods for updating metrics, synchronizing metrics between processes, and logging progress.

    """

    def __init__(self, delimiter="\t"):
        """
        Args:
            delimiter (str): delimiter used to concatenate the output
        """

        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        """
        Update the meter with a new value.

        Args:
            **kwargs (float or int): new value(s) to be added to the meter
        """
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        """
        Gets an attribute from the metric logger.

        If the attribute is a meter (i.e. in `self.meters`), it is returned.
        If the attribute is not a meter, but is in `self.__dict__`, it is
        returned. Otherwise, an `AttributeError` is raised.

        Args:
            attr (str): The name of the attribute to get.

        Returns:
            object: The value of the attribute.

        Raises:
            AttributeError: If the attribute does not exist.
        """
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __str__(self):
        """
        Returns a string representation of the metric logger.

        This method iterates over all meters in the logger, formats their
        string representation, and joins them using the specified delimiter.

        Returns:
            str: Concatenated string of all meters' names and their formatted
                 values, separated by the delimiter.
        """
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {str(meter)}")
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        """
        Synchronize the state of all meters between all processes.

        This method is necessary for metrics that are computed in parallel
        across multiple processes. It ensures that all processes have the
        same state for all meters.
        """
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        """
        Adds a new meter to be tracked by the metric logger.

        Args:
            name (str): The name of the new meter.
            meter (Meter): The meter to be added.
        """

        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        """
        Logs the progress of an iterable at a specified frequency.

        This function iterates over the given iterable, logging metrics and
        estimated time of arrival (ETA) at each specified print frequency.
        It is designed to work with CUDA if available, logging additional
        memory usage statistics.

        Args:
            iterable (iterable): The iterable to iterate over and log.
            print_freq (int): Frequency at which logging should occur.
            header (str, optional): Optional header string to prefix log messages.

        Yields:
            obj: Each element from the iterable.

        Logs:
            Iteration index, ETA, metrics, time per iteration, data loading time,
            and maximum memory allocated (if CUDA is available).
        """
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        if torch.cuda.is_available():
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                    "max mem: {memory:.0f}",
                ]
            )
        else:
            log_msg = self.delimiter.join(
                [header, "[{0" + space_fmt + "}/{1}]", "eta: {eta}", "{meters}", "time: {time}", "data: {data}"]
            )
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    log(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    log(
                        log_msg.format(
                            i, len(iterable), eta=eta_string, meters=str(self), time=str(iter_time), data=str(data_time)
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        log(f"{header} Total time: {total_time_str} ({total_time / len(iterable):.4f} s / it)")


def collate_fn(batch):
    """
    Collate function for DataLoader.

    This function takes a batch of samples as input and collates them into a tuple.
    The output tuple contains the collated data and annotations of the batch.

    Args:
        batch (list): A batch of samples, where each sample is a tuple of (data, annotations).

    Returns:
        tuple: A tuple containing the collated data and annotations of the batch.
    """
    return tuple(zip(*batch))


def mkdir(path):
    """
    Create a directory at the given path if it does not already exist.

    If the directory already exists, this function does nothing.

    Args:
        path (str): The path to create the directory at.

    Raises:
        OSError: If the directory cannot be created.
    """
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    """
    Check if distributed processing is available and initialized.

    This function verifies if the PyTorch distributed package is available
    and if it has been initialized for use. It is typically used to determine
    whether the current environment is set up for distributed training.

    Returns:
        bool: True if distributed processing is both available and initialized,
              False otherwise.
    """
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    """
    Returns the number of processes in the current distributed group.

    This function checks if the distributed processing environment is available
    and initialized. If it is, the function returns the number of processes
    in the current distributed group using `torch.distributed.get_world_size()`.
    If the distributed environment is not available or initialized, the function
    returns 1, indicating a single process.

    Returns:
        int: The number of processes in the current distributed group, or 1 if
             distributed processing is not available or initialized.
    """
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    """
    Returns the rank of the current process in the current distributed group.

    If the distributed processing environment is not available or initialized,
    the function returns 0, indicating the main process.

    Returns:
        int: The rank of the current process in the current distributed group, or 0
             if distributed processing is not available or initialized.
    """
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    """Determine if the current process is the main process.

    Returns:
        bool: True if the current process is the main process, False otherwise.
    """
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    """Call torch.save on the master process

    This function takes the same arguments as torch.save, but only calls it on
    the master process. It is useful for saving models and other objects that
    are only needed on the master process.

    Args:
        *args: The arguments to pass to torch.save
        **kwargs: The keyword arguments to pass to torch.save
    """
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    """Initialize the distributed environment.

    Modified from torchvision/references/detection.

    Args:
        args (dict): The arguments to modify. It should contain the following
            keys:

            - rank (int): The rank of the current process.
            - world_size (int): The number of processes in the current
                distributed group.
            - gpu (int): The GPU to use for the current process.
            - distributed (bool): Whether to use distributed mode or not.
            - dist_url (str): The URL to use for initializing the distributed
                backend.
            - dist_backend (str): The backend to use for distributed training.
                This should be "nccl".

    If environment variables "RANK" and "WORLD_SIZE" are set, use them to
    initialize the distributed environment. Otherwise, set
    `args['distributed']` to False.

    If environment variable "SLURM_PROCID" is set, use it to initialize the
    distributed environment. Otherwise, set `args['distributed']` to False.
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args['rank'] = int(os.environ["RANK"])
        args['world_size'] = int(os.environ["WORLD_SIZE"])
        args['gpu'] = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args['rank'] = int(os.environ["SLURM_PROCID"])
        args['gpu'] = args['rank'] % torch.cuda.device_count()
    else:
        print("Not using distributed mode")
        args['distributed'] = False
        return

    args['distributed'] = True

    torch.cuda.set_device(args['gpu'])
    args['dist_backend'] = "nccl"
    print(f"| distributed init (rank {args['rank']}): {args['dist_url']}", flush=True)
    torch.distributed.init_process_group(
        backend=args['dist_backend'], init_method=args['dist_url'], world_size=args['world_size'], rank=args['rank']
    )
    torch.distributed.barrier()
    setup_for_distributed(args['rank'] == 0)