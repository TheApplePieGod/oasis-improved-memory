from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from itertools import repeat
from pprint import pprint
from tqdm import tqdm
import requests
import argparse
import glob
import json
import os


def download_url(url, save_dir):
    filename = os.path.join(save_dir, url.split('/')[-1])
    if os.path.exists(filename):
        tqdm.write(f"File already exists, skipping: {filename}")
        return filename

    try:
        chunk_size = 1024 * 1024
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(filename, 'wb') as file:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    file.write(chunk)
            return filename
        else:
            tqdm.write(f"Failed to download {url} ({response.status_code})")
            return None
    except Exception as e:
        tqdm.write(f"Error downloading {url}: {e}")
        return None


def validate_id(id, save_dir):
    video_path = os.path.abspath(os.path.join(save_dir, id + ".mp4"))
    json_path = os.path.abspath(os.path.join(save_dir, id + ".jsonl"))
    if not os.path.exists(json_path):
        tqdm.write(f"ID {id} missing .jsonl, removing")
        os.remove(video_path)
    else:
        # Parse json and ensure valid
        try:
            #tqdm.write(f"Loading json {json_path}")
            with open(json_path) as json_file:
                json_lines = json_file.readlines()
                json_data = "[" + ",".join(json_lines) + "]"
                json_data = json.loads(json_data)
        except Exception as e:
            tqdm.write(f"ID {id} invalid .jsonl, removing ({e})")
            os.remove(video_path)
            os.remove(json_path)


def main(args):
    with open(args.index_path, "r") as f:
        index = json.load(f)

    base = index["basedir"]
    paths = index["relpaths"]

    if args.ver is None:
        filtered = paths
    else:
        filtered = []
        for path in paths:
            elems = path.split("/")
            if args.ver in elems:
                filtered.append(path)
    paths = filtered

    if args.limit is not None:
        paths = paths[:args.limit]

    def download(url):
        without_ext = ".".join(url.split(".")[:-1])
        download_url(base + without_ext + ".mp4", args.out_dir)  # Video
        download_url(base + without_ext + ".jsonl", args.out_dir)  # Actions

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        results = list(tqdm(executor.map(download, paths), total=len(paths)))

    # ===== Validate =====

    unique_ids = glob.glob(os.path.join(args.out_dir, "*.mp4"))
    unique_ids = list(set([os.path.basename(x).split(".")[0] for x in unique_ids]))
    print(f"Found {len(unique_ids)} unique videos, validating...")
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        results = list(tqdm(executor.map(validate_id, unique_ids, repeat(args.out_dir)), total=len(unique_ids)))


if __name__ == "__main__":
    parse = argparse.ArgumentParser()

    parse.add_argument(
        "--index-path",
        type=str,
        help="Path of the index file for the data to download",
        default=None,
    )
    parse.add_argument(
        "--out-dir",
        type=str,
        help="Path to output folder",
        default="data",
    )
    parse.add_argument(
        "--ver",
        type=str,
        help="Specific version to restrict",
        default=None,
    )
    parse.add_argument(
        "--num-workers",
        type=int,
        help="Number of concurrent download workers",
        default=8,
    )
    parse.add_argument(
        "--limit",
        type=int,
        help="Max videos to download",
        default=None,
    )

    args = parse.parse_args()
    print("train args:")
    pprint(vars(args))
    main(args)
