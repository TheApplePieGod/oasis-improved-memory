import torch
from torch.utils.data import Dataset
from einops import rearrange
from actions import process_json_action, one_hot_actions
from tqdm import tqdm
import numpy as np
import random
import torch
import glob
import json
import av
import os


def oasis_dataset_collate(batch):
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
        load_actions=True
    ):
        assert max_seq_len > 0

        self.data_dir = data_dir
        self.image_size = image_size
        self.max_seq_len = max_seq_len
        self.load_actions = load_actions

        self.datapoints = []
        unique_ids = glob.glob(os.path.join(data_dir, "*.mp4"))
        unique_ids = list(set([os.path.basename(x).split(".")[0] for x in unique_ids]))
        print("Loading dataset metadata")
        for id in tqdm(unique_ids):
            video_path = os.path.abspath(os.path.join(self.data_dir, id + ".mp4"))
            json_path = os.path.abspath(os.path.join(self.data_dir, id + ".jsonl"))

            try:
                with av.open(video_path) as video:
                    video_stream = video.streams.video[0]
                    frame_count = video_stream.frames  # Total number of frames
            except:
                continue

            #if frame_count < max_seq_len:
            #    continue

            self.datapoints.append({
                "id": id,
                "video_path": video_path,
                "json_path": json_path,
                "frame_count": int(frame_count)
            })

        if max_datapoints is not None:
            assert max_datapoints > 0
            self.datapoints = self.datapoints[:max_datapoints]

        print(f"Dataset initialized with {len(self.datapoints)} datapoints")

    def __getitem__(self, idx):
        # T C H W
        #return torch.rand((self.max_seq_len, 3, 360, 640))

        data = self.datapoints[idx]

        seq_len = min(self.max_seq_len, data["frame_count"])
        start_frame = random.randint(0, data["frame_count"] - seq_len)

        if self.load_actions:
            with open(data["json_path"]) as json_file:
                json_lines = json_file.readlines()
                json_data = "[" + ",".join(json_lines) + "]"
                json_data = json.loads(json_data)
        else:
            json_data = [0] * seq_len

        frames = []
        actions = []
        with av.open(data["video_path"]) as video:
            # Rough estimate of the desired frame. Could do some more math
            # and decoding to get the exact frame, but it's not that important
            # https://github.com/PyAV-Org/PyAV/discussions/1113
            if not self.load_actions:
                stream = video.streams.video[0]
                seek_sec = start_frame / stream.average_rate
                seek_ts = int(seek_sec / stream.time_base)
                video.seek(seek_ts, stream=stream)

            frame_iter = video.decode(video=0)
            for i in range(len(json_data)):
                # TODO: optimize?
                if self.load_actions and i < start_frame:
                    next(frame_iter)
                    continue

                frame = next(frame_iter)

                if self.load_actions:
                    action = json_data[i]
                    action, is_null = process_json_action(action)

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
