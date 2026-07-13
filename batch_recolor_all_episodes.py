from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from batch_recolor_episode import _parse_cameras, recolor_episode
from add_notation import DEFAULT_PINK_BLOCK_PROMPTS


def episode_sort_key(path: Path) -> tuple[int, str]:
    name = path.name
    if name.startswith("episode"):
        suffix = name[len("episode") :]
        if suffix.isdigit():
            return int(suffix), name
    return 10**9, name


def iter_episode_dirs(pick_place_root: Path, episodes: Optional[Sequence[str]] = None) -> list[Path]:
    if episodes:
        out = []
        for episode in episodes:
            name = episode if episode.startswith("episode") else f"episode{episode}"
            path = pick_place_root / name
            if not path.exists():
                raise FileNotFoundError(f"Episode directory not found: {path}")
            out.append(path)
        return out

    return sorted(
        (path for path in pick_place_root.iterdir() if path.is_dir() and path.name.startswith("episode")),
        key=episode_sort_key,
    )


def batch_recolor_all_episodes(
    pick_place_root: Path,
    output_root: Path,
    target_color: str = "blue",
    *,
    episodes: Optional[Sequence[str]] = None,
    cameras: Optional[Sequence[str]] = None,
    checkpoint: Optional[Path] = None,
    device: Optional[str] = None,
    threshold: float = 0.5,
    alpha: float = 0.9,
    preserve_luminance: bool = True,
    all_masks: bool = False,
    overwrite: bool = False,
    copy_metadata: bool = False,
    limit_per_episode: Optional[int] = None,
    stop_on_error: bool = False,
) -> tuple[int, int, int]:
    episode_dirs = iter_episode_dirs(pick_place_root, episodes=episodes)
    if not episode_dirs:
        raise RuntimeError(f"No episode directories found under {pick_place_root}")

    output_root.mkdir(parents=True, exist_ok=True)
    total_ok = 0
    total_failed = 0
    completed_episodes = 0
    prompts = DEFAULT_PINK_BLOCK_PROMPTS

    print(f"Found {len(episode_dirs)} episodes under {pick_place_root}")
    print(f"Output root: {output_root}")
    for ep_idx, episode_dir in enumerate(episode_dirs, start=1):
        output_episode_dir = output_root / episode_dir.name
        print(f"\n=== [{ep_idx}/{len(episode_dirs)}] {episode_dir.name} -> {output_episode_dir} ===")
        try:
            ok, failed = recolor_episode(
                episode_dir,
                output_episode_dir,
                target_color,
                cameras=cameras,
                prompts=prompts,
                checkpoint=checkpoint,
                device=device,
                threshold=threshold,
                alpha=alpha,
                preserve_luminance=preserve_luminance,
                all_masks=all_masks,
                skip_existing=not overwrite,
                copy_metadata=copy_metadata,
                limit=limit_per_episode,
            )
        except Exception as exc:
            if stop_on_error:
                raise
            ok, failed = 0, 1
            print(f"Episode failed: {episode_dir}: {exc}")

        total_ok += ok
        total_failed += failed
        completed_episodes += 1
        print(f"Episode summary: success={ok}, failed={failed}")

    return completed_episodes, total_ok, total_failed


def _parse_episodes(values: Optional[list[str]]) -> Optional[list[str]]:
    if not values:
        return None
    out: list[str] = []
    for value in values:
        out.extend(part.strip() for part in value.split(",") if part.strip())
    return out or None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recolor the pink small block in RGB images for all pick_place episodes."
    )
    parser.add_argument(
        "--pick-place-root",
        type=Path,
        default=Path("pick_place"),
        help="Root directory containing episode* folders.",
    )
    parser.add_argument(
        "target_color",
        nargs="?",
        default="blue",
        help="Target color name, e.g. blue, green, cyan, 官方蓝色.",
    )
    parser.add_argument(
        "-o",
        "--output-root",
        type=Path,
        default=None,
        help="Output root. Defaults to tmp/pick_place_<target_color>.",
    )
    parser.add_argument(
        "--episode",
        action="append",
        default=None,
        help="Episode id/name to process. Can be repeated or comma-separated. Defaults to all episodes.",
    )
    parser.add_argument(
        "--camera",
        action="append",
        default=None,
        help="Camera name to process. Can be repeated or comma-separated. Defaults to all color cameras.",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--all-masks", action="store_true")
    parser.add_argument("--no-preserve-luminance", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output RGB images.")
    parser.add_argument(
        "--copy-metadata",
        action="store_true",
        help="Copy non-image files too. Default is RGB-only output.",
    )
    parser.add_argument(
        "--limit-per-episode",
        type=int,
        default=None,
        help="Process only the first N RGB images in each episode.",
    )
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args()

    output_root = args.output_root
    if output_root is None:
        output_root = Path("tmp") / f"pick_place_{args.target_color}"

    episodes = _parse_episodes(args.episode)
    cameras = _parse_cameras(args.camera)
    completed, ok, failed = batch_recolor_all_episodes(
        args.pick_place_root,
        output_root,
        args.target_color,
        episodes=episodes,
        cameras=cameras,
        checkpoint=args.checkpoint,
        device=args.device,
        threshold=args.threshold,
        alpha=args.alpha,
        preserve_luminance=not args.no_preserve_luminance,
        all_masks=args.all_masks,
        overwrite=args.overwrite,
        copy_metadata=args.copy_metadata,
        limit_per_episode=args.limit_per_episode,
        stop_on_error=args.stop_on_error,
    )
    print(
        f"\nDone. episodes={completed}, success_images={ok}, "
        f"failed_images={failed}, output_root={output_root}"
    )


if __name__ == "__main__":
    main()
