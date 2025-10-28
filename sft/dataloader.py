from torch.utils.data import DataLoader
import os

def func(batch):
    return batch[0]


def worker_init_fn(worker_id):
    os.sched_setaffinity(0, range(192))


def infinite_dataloader(dataset, num_workers, pin_memory, prefetch_factor):
    def get_iter(epoch):
        dataset.set_epoch(epoch)
        dataloader = DataLoader(
            dataset,
            num_workers=num_workers,
            batch_size=1,
            collate_fn=func,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            worker_init_fn=worker_init_fn
        )
        iterator = iter(dataloader)
        return iterator

    epoch = 0
    iterator = get_iter(epoch)

    while True:
        try:
            data = next(iterator)
        except StopIteration:
            epoch += 1
            iterator = get_iter(epoch)
            data = next(iterator)
        yield data
