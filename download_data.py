from concurrent.futures import ThreadPoolExecutor
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
        print(f"File already exists, skipping: {filename}")
        return filename

    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:

            with open(filename, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            return filename
        else:
            print(f"Failed to download {url}")
            return None
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return None


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

    # Validate
    unique_ids = glob.glob(os.path.join(args.out_dir, "*.mp4"))
    unique_ids = list(set([os.path.basename(x).split(".")[0] for x in unique_ids]))
    for id in unique_ids:
        video_path = os.path.abspath(os.path.join(args.out_dir, id + ".mp4"))
        json_path = os.path.abspath(os.path.join(args.out_dir, id + ".jsonl"))
        if not os.path.exists(json_path):
            print(f"ID {id} missing .jsonl, removing")
            os.remove(video_path)


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
