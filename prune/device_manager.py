import torch

class DeviceManager:
    def __init__(self, compute_device, storage_device, aggregation_device):
        """
        Manage device allocation for different operations.

        Args:
            compute_device (str): Device for forward pass and initial computations
            storage_device (str): Device for storing intermediate activations
            aggregation_device (str): Device for final aggregation
        """
        self.compute_device = torch.device(compute_device)
        self.storage_device = torch.device(storage_device)
        self.aggregation_device = torch.device(aggregation_device)

    def validate_devices(self):
        """Validate that all specified devices are available."""
        num_gpus = torch.cuda.device_count()
        for device in [self.compute_device, self.storage_device, self.aggregation_device]:
            if device.type == "cuda":
                device_idx = int(str(device).split(":")[-1])
                if device_idx >= num_gpus:
                    raise ValueError(f"Device {device} not available. System has {num_gpus} GPUs.")

    def get_memory_status(self):
        """Get available memory for all devices."""
        memory_status = {}
        for i in range(torch.cuda.device_count()):
            torch.cuda.memory_stats(i)
            free_memory = torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i)
            memory_status[f"cuda:{i}"] = free_memory
        return memory_status
