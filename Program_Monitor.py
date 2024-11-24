from tqdm import tqdm

def progress_bar(task_name, current, total):
    with tqdm(total=total, desc=task_name) as pbar:
        for _ in range(current):
            pbar.update(1)

def monitor_progress(iterable, description="Processing"):
    for item in tqdm(iterable, desc=description):
        yield item