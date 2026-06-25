import torch
import torch.distributed as dist
from torch.utils.data.sampler import Sampler
from collections import defaultdict

class DistributedAspectRatioBucketSampler(Sampler):
    def __init__(self, dataset, batch_size, num_replicas=None, rank=None, drop_last=False, shuffle=True):
        super().__init__(dataset)
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.epoch = 0

        if num_replicas is None:
            if dist.is_available() and dist.is_initialized():
                num_replicas = dist.get_world_size()
            else:
                num_replicas = 1

        if rank is None:
            if dist.is_available() and dist.is_initialized():
                rank = dist.get_rank()
            else:
                rank = 0
            
        self.num_replicas = num_replicas
        self.rank = rank
        self.buckets = self._get_buckets()

    def _get_buckets(self):
        buckets = defaultdict(list)
        for i in range(len(self.dataset)):
            resolution = self.dataset.get_resolution(i)
            buckets[resolution].append(i)
        return buckets

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)
        
        bucket_keys = list(self.buckets.keys())
        
        if self.shuffle:
            indices = torch.randperm(len(bucket_keys), generator=g).tolist()
            bucket_keys = [bucket_keys[i] for i in indices]
            
        all_batches = []
        
        for key in bucket_keys:
            indices = self.buckets[key]
            if self.shuffle:
                indices = torch.tensor(indices)[torch.randperm(len(indices), generator=g)].tolist()
            
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i : i + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    all_batches.append(batch)

        if self.shuffle:
            batch_indices = torch.randperm(len(all_batches), generator=g).tolist()
            all_batches = [all_batches[i] for i in batch_indices]

        num_batches_per_replica = len(all_batches) // self.num_replicas
        total_batches_needed = num_batches_per_replica * self.num_replicas
        all_batches = all_batches[:total_batches_needed]
        
        my_batches = all_batches[self.rank : total_batches_needed : self.num_replicas]
        
        return iter(my_batches)

    def __len__(self):
        total_count = 0
        for indices in self.buckets.values():
            if self.drop_last:
                total_count += len(indices) // self.batch_size
            else:
                total_count += (len(indices) + self.batch_size - 1) // self.batch_size
        return total_count // self.num_replicas

    def set_epoch(self, epoch):
        self.epoch = epoch