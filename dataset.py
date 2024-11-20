from torch.utils.data import Dataset
from einops import rearrange
import numpy as np
import random
import torch
import glob
import json
import av
import os

class OasisDataset(Dataset):
    def __init__(
        self,
        data_dir,
        image_size,
        max_seq_len,
        max_datapoints=None
    ):
        assert max_seq_len > 0

        self.data_dir = data_dir
        self.image_size = image_size
        self.max_seq_len = max_seq_len

        self.datapoints = []
        unique_ids = glob.glob(os.path.join(data_dir, "*.mp4"))
        unique_ids = list(set([os.path.basename(x).split(".")[0] for x in unique_ids]))
        for id in unique_ids:
            video_path = os.path.abspath(os.path.join(self.data_dir, id + ".mp4"))
            json_path = os.path.abspath(os.path.join(self.data_dir, id + ".jsonl"))

            try:
                with av.open(video_path) as video:
                    video_stream = video.streams.video[0]
                    frame_count = video_stream.frames  # Total number of frames
            except:
                continue

            if frame_count < max_seq_len:
                continue

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

        seq_len = self.max_seq_len
        start_frame = random.randint(1, data["frame_count"] - seq_len) - 1

        frames = []
        with av.open(data["video_path"]) as video:
            stream = video.streams.video[0]

            # Rough estimate of the desired frame. Could do some more math
            # and decoding to get the exact frame, but it's not that important
            # https://github.com/PyAV-Org/PyAV/discussions/1113
            seek_sec = start_frame / stream.average_rate
            seek_ts = int(seek_sec / stream.time_base)
            video.seek(seek_ts, stream=stream)

            for frame in video.decode(video=0):
                frame = frame.to_ndarray(
                    width=self.image_size[0],
                    height=self.image_size[1],
                    format="rgb24"
                )
                frames.append([frame])

                if len(frames) >= seq_len:
                    break

        frames = np.vstack(frames)
        frames = torch.from_numpy(frames)
        frames = rearrange(frames, "t h w c -> t c h w")
        frames = frames.float() / 255.0  # Normalize to [0, 1]
        frames = frames * 2.0 - 1.0  # Normalize to [-1, 1]
        return frames

    def __len__(self):
        return len(self.datapoints)
