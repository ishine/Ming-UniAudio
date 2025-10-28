import json
import random
import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset
from transformers import AutoProcessor
import os

from sft.dataloader import infinite_dataloader
from sft.processors import DynamicBatch, Padding, SampleBuilder, Sort


class Processor(IterableDataset):

    def __init__(self, source, f, *args, **kw):
        assert callable(f)
        self.source = source
        self.f = f
        self.args = args
        self.kw = kw

    def set_epoch(self, epoch):
        self.source.set_epoch(epoch)

    def __iter__(self):
        """ Return an iterator over the source dataset processed by the
            given processor.
        """
        assert self.source is not None
        assert callable(self.f)
        return self.f(iter(self.source), *self.args, **self.kw)

    def apply(self, f):
        assert callable(f)
        return Processor(self, f, *self.args, **self.kw)


class SftDataset(IterableDataset):
    def __init__(self, data_jsonl_file):
        self.data = self.__read_data_jsonl(data_jsonl_file)

        self.update()

    def __read_data_jsonl(self, file_path):
        data = []
        with open(file_path, 'r') as f:
            for line in f:
                tmp = json.loads(line.strip())
                data.append(tmp)
        return data
            
    def set_epoch(self, epoch):
        self.epoch = epoch

    def update(self):
        assert dist.is_available()
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            self.worker_id = 0
            self.num_workers = 1
        else:
            self.worker_id = worker_info.id
            self.num_workers = worker_info.num_workers

    def __iter__(self):
        self.update()
        random.Random(self.epoch).shuffle(self.data)
        assert len(self.data) >= self.world_size * self.num_workers
        data = self.data[self.rank::self.world_size][self.worker_id::self.num_workers]
        for item in data:
            yield item

def build_dataset(
    data_jsonl_file, 
    tokenizer=False,
    sr=16000,
    patch_size=5,
    hop_size=320,
    max_frames_in_batch=1000,
    buffer_size=100
):
    dataset = SftDataset(data_jsonl_file)
    dataset = Processor(dataset, SampleBuilder(tokenizer, sr=sr, patch_size=patch_size, hop_size=hop_size))
    dataset = Processor(dataset, Sort(buffer_size=buffer_size))
    dataset = Processor(dataset, DynamicBatch(max_frames_in_batch=max_frames_in_batch))
    dataset = Processor(dataset, Padding(tokenizer=tokenizer))
    return dataset


if __name__ == '__main__':
    import os

    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    processor = AutoProcessor.from_pretrained(".", trust_remote_code=True)
    tokenizer = processor.tokenizer
        
    dataset = build_dataset('sft/data/tts.jsonl', tokenizer=tokenizer)
    dataset.set_epoch(0)
    
    dataloader = infinite_dataloader(dataset=dataset, num_workers=2, pin_memory=True, prefetch_factor=2)
    for item in dataloader:
        print(rank, item)
        break