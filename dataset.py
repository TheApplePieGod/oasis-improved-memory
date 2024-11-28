from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from torch.utils.data import Dataset
from einops import rearrange
from actions import process_json_action, one_hot_actions
from tqdm import tqdm
import numpy as np
import msgspec
import random
import torch
import time
import glob
import av
import os


def print_timing(name, start, end):
    tqdm.write(f"{name} took {(end-start)*1000:.2f}ms")


def oasis_dataset_collate(batch):
    # Remove null elements (empty sequences)
    batch = [b for b in batch if b is not None]

    # Ensure that the batch has the same sequence lengths if some
    # were cut short
    min_seq = min(x[0].shape[0] for x in batch)
    return (
        torch.stack([x[0][:min_seq] for x in batch]),
        torch.stack([x[1][:min_seq] for x in batch]),
    )


class OasisDataset(Dataset):
    def __init__(
        self,
        data_dir,
        image_size,
        max_seq_len,
        max_datapoints=None,
        load_actions=True,
        preload_json=False
    ):
        assert max_seq_len > 0

        self.data_dir = data_dir
        self.image_size = image_size
        self.max_seq_len = max_seq_len
        self.load_actions = load_actions
        self.preload_json = preload_json

        self.datapoints = []
        unique_ids = glob.glob(os.path.join(data_dir, "*.mp4"))
        unique_ids = list(set([os.path.basename(x).split(".")[0] for x in unique_ids]))
        print("Loading dataset metadata")

        def load_id(id):
            video_path = os.path.abspath(os.path.join(self.data_dir, id + ".mp4"))
            json_path = os.path.abspath(os.path.join(self.data_dir, id + ".jsonl"))

            try:
                with av.open(video_path) as video:
                    video_stream = video.streams.video[0]
                    frame_count = video_stream.frames  # Total number of frames
            except:
                return

            # Could be a better heuristic but exclude shorter videos 
            if frame_count < 256:
                return

            json_data = None
            if self.preload_json:
                json_data = self.load_json(json_path)

            return {
                "id": id,
                "video_path": video_path,
                "json_path": json_path,
                "json_data": json_data,
                "frame_count": int(frame_count)
            }

        # If we are preloading the json, do it in parallel since it is slow. Otherwise, the overhead is too high
        # so don't parallelize it
        if self.preload_json:
            with ThreadPoolExecutor() as executor:
                self.datapoints = list(tqdm(executor.map(load_id, unique_ids), total=len(unique_ids)))
                self.datapoints = [d for d in self.datapoints if d is not None]
        else:
            self.datapoints = []
            for id in tqdm(unique_ids):
                loaded = load_id(id)
                if loaded is not None:
                    self.datapoints.append(loaded)

        if max_datapoints is not None:
            assert max_datapoints > 0
            self.datapoints = self.datapoints[:max_datapoints]

        print(f"Dataset initialized with {len(self.datapoints)} datapoints")

    def load_json(self, path):
        with open(path) as f:
            return [process_json_action(msgspec.json.decode(l)) for l in f]

    def __getitem__(self, idx):
        # T C H W
        #return torch.rand((self.max_seq_len, 3, 360, 640))

        data = self.datapoints[idx]

        seq_len = min(self.max_seq_len, data["frame_count"])
        start_frame = random.randint(0, data["frame_count"] - seq_len)
        start_time = time.time()

        if self.load_actions:
            if self.preload_json:
                json_data = data["json_data"]
            else:
                json_data = self.load_json(data["json_path"])
        else:
            json_data = [0] * data["frame_count"]

        frames = []
        actions = []
        with av.open(data["video_path"]) as video:
            # Rough estimate of the desired frame. Could do some more math
            # and decoding to get the exact frame, but it's not that important
            # https://github.com/PyAV-Org/PyAV/discussions/1113
            stream = video.streams.video[0]
            seek_sec = start_frame / stream.average_rate
            seek_ts = round(seek_sec / stream.time_base)
            video.seek(seek_ts, stream=stream)
            
            # Compute the actual start frame we seeked to
            frame_iter = video.decode(video=0)
            frame = next(frame_iter)
            start_frame = int(frame.pts * stream.time_base * stream.average_rate)

            for i in range(start_frame, len(json_data)):
                if self.load_actions:
                    action, is_null = json_data[i]

                    # If nothing happened, skip to the next frame
                    if is_null:
                        continue

                    actions.append(action)

                frame = frame.to_ndarray(
                    width=self.image_size[0],
                    height=self.image_size[1],
                    format="rgb24"
                )
                frames.append([frame])

                if len(frames) >= seq_len:
                    break

                frame = next(frame_iter)

        # If the sequence is empty, return None rather than an empty tensor
        if not frames:
            return None

        end_time = time.time()
        #print_timing("Loading frame", start_time, end_time)

        actions = one_hot_actions(actions)
        actions = torch.cat([torch.zeros_like(actions[:1]), actions], dim=0) # prepend null action
        frames = np.vstack(frames)
        frames = torch.from_numpy(frames)
        frames = rearrange(frames, "t h w c -> t c h w")
        frames = frames.float() / 255.0  # Normalize to [0, 1]
        frames = frames * 2.0 - 1.0  # Normalize to [-1, 1]
        return frames, actions

    def __len__(self):
        return len(self.datapoints)
