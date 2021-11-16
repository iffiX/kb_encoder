import os
import functools
import pathlib
import gzip
import zipfile
import shutil
import requests
from tqdm.auto import tqdm


def open_file_with_create_directories(path, mode):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, mode=mode)


def download_to(url, path):
    # https://stackoverflow.com/questions/
    # 37573483/progress-bar-while-download-file-over-http-with-requests
    r = requests.get(url, stream=True, allow_redirects=True)
    if r.status_code != 200:
        r.raise_for_status()  # Will only raise for 4xx codes, so...
        raise RuntimeError(f"Request to {url} returned status code {r.status_code}")
    file_size = int(r.headers.get("Content-Length", 0))

    path = pathlib.Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    desc = "(Unknown total file size)" if file_size == 0 else ""
    r.raw.read = functools.partial(
        r.raw.read, decode_content=True
    )  # Decompress if needed
    with tqdm.wrapattr(r.raw, "read", total=file_size, desc=desc) as r_raw:
        with path.open("wb") as f:
            shutil.copyfileobj(r_raw, f)


def decompress_gz(path_from, path_to):
    with gzip.open(path_from, "rb") as f_in:
        with open(path_to, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)


def decompress_zip(path_from, dir_path_to):
    with zipfile.ZipFile(path_from, "r") as zip_ref:
        zip_ref.extractall(dir_path_to)
