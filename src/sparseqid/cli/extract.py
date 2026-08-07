# SPDX-License-Identifier: Apache-2.0

"""Extract the native-540 JPEG cache used by SparseQID."""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from pathlib import Path

import cv2
import decord


def parser(prog: str | None = None) -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog=prog, description=__doc__)
    result.add_argument("--split", default="val")
    result.add_argument("--scenes", nargs="+", required=True)
    result.add_argument("--every", type=int, default=1)
    result.add_argument("--scale", type=float, default=0.5)
    result.add_argument("--quality", type=int, default=90)
    result.add_argument("--num-frames", "--n-frames", type=int, default=9000)
    result.add_argument(
        "--data-root", "--root", default=os.environ.get("AICITY26_DATA", "data/MTMC_Tracking_2026")
    )
    result.add_argument(
        "--cache-root",
        "--cache",
        default=os.environ.get("AICITY26_CACHE", "data/aicity2026_frames_540"),
    )
    return result


def main(argv: Sequence[str] | None = None, *, prog: str | None = None) -> None:
    args = parser(prog).parse_args(argv)
    encoding = [cv2.IMWRITE_JPEG_QUALITY, args.quality]
    for scene in args.scenes:
        videos = sorted((Path(args.data_root) / args.split / scene / "videos").glob("*.mp4"))
        if not videos:
            raise FileNotFoundError(f"no videos found for {scene}")
        print(f"{scene}: {len(videos)} cameras", flush=True)
        for video in videos:
            output = Path(args.cache_root) / args.split / scene / video.stem
            output.mkdir(parents=True, exist_ok=True)
            reader = decord.VideoReader(str(video), num_threads=1)
            written = 0
            for frame_id in range(0, min(len(reader), args.num_frames), args.every):
                destination = output / f"{frame_id:06d}.jpg"
                if destination.exists():
                    continue
                image = cv2.cvtColor(reader[frame_id].asnumpy(), cv2.COLOR_RGB2BGR)
                if args.scale != 1:
                    image = cv2.resize(
                        image,
                        None,
                        fx=args.scale,
                        fy=args.scale,
                        interpolation=cv2.INTER_AREA,
                    )
                if not cv2.imwrite(str(destination), image, encoding):
                    raise OSError(f"failed to write {destination}")
                written += 1
            print(f"  {video.stem}: wrote {written}", flush=True)


if __name__ == "__main__":
    main()
