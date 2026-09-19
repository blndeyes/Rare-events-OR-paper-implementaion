"""Deterministic editing of ellipse-conditioning trajectories.

The paper states that captured waypoints are interpolated over the complete video,
but does not specify timing or simplification semantics.  This implementation uses
Ramer-Douglas-Peucker simplification followed by constant-speed, arc-length linear
resampling.  The default ``replace`` mode makes the selected centroid follow that
97-point path; ``offset`` adds the path displacement to the source centroid track.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from or_video_reproduction import __version__
from or_video_reproduction.data.clips import (
    OUTPUT_FPS,
    OUTPUT_FRAME_COUNT,
    TARGET_HEIGHT,
    TARGET_WIDTH,
)

from .ellipse import Ellipse
from .palette import PAPER_36_PALETTE
from .render import RenderInstance, render_conditioning

COORDINATE_CONVENTION = (
    "image_xy_pixels: origin at top-left; +x right; +y down; coordinates refer to "
    "the 1024x768 rendered conditioning frame"
)
INTERPOLATION_METHOD = (
    "Ramer-Douglas-Peucker simplification in pixel space, then piecewise-linear "
    "constant-speed resampling by cumulative Euclidean arc length at 97 inclusive samples"
)


def _finite_point(value: object, *, field: str) -> tuple[float, float]:
    if isinstance(value, Mapping):
        x, y = value.get("x"), value.get("y")
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        x, y = value
    else:
        raise ValueError(f"{field} must be an [x, y] pair or an object with x and y")
    if isinstance(x, bool) or isinstance(y, bool):
        raise ValueError(f"{field} coordinates must be finite numbers")  # noqa: TRY004
    try:
        point = float(x), float(y)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} coordinates must be finite numbers") from error
    if not all(math.isfinite(item) for item in point):
        raise ValueError(f"{field} coordinates must be finite numbers")
    return point


def load_trajectory(path: Path) -> dict[str, object]:
    """Load and strictly validate a reproducible trajectory JSON file."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Malformed trajectory JSON: {error.msg}") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("Trajectory must be an object with schema_version 1")
    raw = payload.get("waypoints")
    if not isinstance(raw, list) or len(raw) < 2:
        raise ValueError("Trajectory waypoints must contain at least two points")
    waypoints = [
        _finite_point(value, field=f"waypoints[{index}]") for index, value in enumerate(raw)
    ]
    mode = payload.get("mode", "replace")
    if mode not in {"replace", "offset"}:
        raise ValueError("Trajectory mode must be 'replace' or 'offset'")
    epsilon = payload.get("simplify_epsilon_pixels", 2.0)
    if isinstance(epsilon, bool):
        raise ValueError(  # noqa: TRY004
            "simplify_epsilon_pixels must be a finite non-negative number"
        )
    try:
        epsilon = float(epsilon)
    except (TypeError, ValueError) as error:
        raise ValueError("simplify_epsilon_pixels must be a finite non-negative number") from error
    if not math.isfinite(epsilon) or epsilon < 0:
        raise ValueError("simplify_epsilon_pixels must be a finite non-negative number")
    return {
        "waypoints": waypoints,
        "mode": mode,
        "simplify_epsilon_pixels": epsilon,
    }


def load_sequence_metadata(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    frames = payload.get("frames") if isinstance(payload, dict) else None
    shape = payload.get("shape") if isinstance(payload, dict) else None
    if not isinstance(frames, list) or len(frames) != OUTPUT_FRAME_COUNT:
        raise ValueError(f"Ellipse metadata must contain exactly {OUTPUT_FRAME_COUNT} frames")
    if shape != [OUTPUT_FRAME_COUNT, TARGET_HEIGHT, TARGET_WIDTH]:
        raise ValueError(
            "Ellipse metadata must have shape "
            f"[{OUTPUT_FRAME_COUNT}, {TARGET_HEIGHT}, {TARGET_WIDTH}]"
        )
    if payload.get("fps") != OUTPUT_FPS:
        raise ValueError(f"Ellipse metadata must use {OUTPUT_FPS} fps")
    for expected_index, frame in enumerate(frames):
        if not isinstance(frame, dict) or frame.get("frame_index") != expected_index:
            raise ValueError("Ellipse metadata frames must be ordered and zero-indexed")
        if not isinstance(frame.get("instances"), list):
            raise ValueError(f"Frame {expected_index} has no instances list")  # noqa: TRY004
    return payload


def _ellipse(record: Mapping[str, object]) -> Ellipse:
    value = record.get("ellipse")
    if not isinstance(value, Mapping):
        raise ValueError("Instance is missing ellipse metadata")  # noqa: TRY004
    try:
        ellipse = Ellipse(
            center_x=float(value["center_x"]),
            center_y=float(value["center_y"]),
            major_diameter=float(value["major_diameter"]),
            minor_diameter=float(value["minor_diameter"]),
            angle_degrees=float(value["angle_degrees"]),
            source_pixels=int(value["source_pixels"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Instance has malformed ellipse metadata") from error
    if (
        not all(
            math.isfinite(value)
            for value in (
                ellipse.center_x,
                ellipse.center_y,
                ellipse.major_diameter,
                ellipse.minor_diameter,
                ellipse.angle_degrees,
            )
        )
        or ellipse.major_diameter <= 0
        or ellipse.minor_diameter <= 0
    ):
        raise ValueError("Instance has invalid ellipse geometry")
    return ellipse


def point_inside_ellipse(point: tuple[float, float], ellipse: Ellipse) -> bool:
    x, y = point
    theta = math.radians(ellipse.angle_degrees)
    dx, dy = x - ellipse.center_x, y - ellipse.center_y
    along_major = dx * math.cos(theta) + dy * math.sin(theta)
    along_minor = -dx * math.sin(theta) + dy * math.cos(theta)
    return (along_major / (ellipse.major_diameter / 2.0)) ** 2 + (
        along_minor / (ellipse.minor_diameter / 2.0)
    ) ** 2 <= 1.0 + 1e-12


def select_instance(
    metadata: Mapping[str, object],
    *,
    instance_id: str | None = None,
    click: tuple[float, float] | None = None,
) -> tuple[str, str]:
    """Select by exact instance key or by the topmost ellipse under a click."""

    if (instance_id is None) == (click is None):
        raise ValueError("Choose exactly one of instance_id or click")
    instances = metadata["frames"][0]["instances"]
    if instance_id is not None:
        matches = [row for row in instances if row.get("key") == instance_id]
        if len(matches) != 1:
            raise ValueError(f"Instance ID is not uniquely present in frame zero: {instance_id!r}")
        return str(matches[0]["key"]), str(matches[0]["class_name"])

    assert click is not None
    candidates = [row for row in instances if point_inside_ellipse(click, _ellipse(row))]
    if not candidates:
        raise ValueError(f"No ellipse contains click ({click[0]:.3f}, {click[1]:.3f})")

    smaller_is_nearer = metadata.get("depth_direction") == "smaller_is_nearer"

    def drawing_key(row: Mapping[str, object]) -> tuple[float, str, str]:
        raw = row.get("raw_relative_depth_mean")
        if raw is None:
            depth_key = float("-inf")
        else:
            depth_key = -float(raw) if smaller_is_nearer else float(raw)
        return depth_key, str(row.get("class_name")), str(row.get("key"))

    selected = max(candidates, key=drawing_key)
    return str(selected["key"]), str(selected["class_name"])


def list_instances(
    metadata: Mapping[str, object], *, frame_index: int = 0
) -> list[dict[str, object]]:
    """Return deterministic, JSON-ready ellipse details for one frame."""

    frames = metadata.get("frames")
    if not isinstance(frames, list) or not 0 <= frame_index < len(frames):
        raise ValueError(f"Frame index is outside the metadata range: {frame_index}")
    frame = frames[frame_index]
    if not isinstance(frame, Mapping) or not isinstance(frame.get("instances"), list):
        raise ValueError(f"Frame {frame_index} has no instances list")  # noqa: TRY004

    rows: list[dict[str, object]] = []
    for record in frame["instances"]:
        if not isinstance(record, Mapping):
            raise ValueError(  # noqa: TRY004
                f"Frame {frame_index} contains a malformed instance"
            )
        ellipse = _ellipse(record)
        rows.append(
            {
                "instance_id": str(record.get("key")),
                "class_name": str(record.get("class_name")),
                "centroid": [ellipse.center_x, ellipse.center_y],
                "major_diameter": ellipse.major_diameter,
                "minor_diameter": ellipse.minor_diameter,
                "angle_degrees": ellipse.angle_degrees,
                "raw_relative_depth_mean": record.get("raw_relative_depth_mean"),
            }
        )
    return rows


def _perpendicular_distance(
    point: tuple[float, float], start: tuple[float, float], end: tuple[float, float]
) -> float:
    p = np.asarray(point, dtype=np.float64)
    a = np.asarray(start, dtype=np.float64)
    b = np.asarray(end, dtype=np.float64)
    segment = b - a
    denominator = float(segment @ segment)
    if denominator == 0:
        return float(np.linalg.norm(p - a))
    projection = a + min(1.0, max(0.0, float((p - a) @ segment) / denominator)) * segment
    return float(np.linalg.norm(p - projection))


def simplify_waypoints(
    waypoints: Sequence[tuple[float, float]], epsilon: float
) -> list[tuple[float, float]]:
    """Deterministic Ramer-Douglas-Peucker simplification."""

    if len(waypoints) < 2:
        raise ValueError("At least two waypoints are required")
    if epsilon < 0 or not math.isfinite(epsilon):
        raise ValueError("Simplification epsilon must be finite and non-negative")
    if epsilon == 0:
        return list(waypoints)
    start, end = waypoints[0], waypoints[-1]
    distances = [_perpendicular_distance(point, start, end) for point in waypoints[1:-1]]
    if not distances:
        return [start, end]
    greatest = max(distances)
    if greatest <= epsilon:
        return [start, end]
    split = 1 + distances.index(greatest)
    left = simplify_waypoints(waypoints[: split + 1], epsilon)
    right = simplify_waypoints(waypoints[split:], epsilon)
    return left[:-1] + right


def interpolate_waypoints(
    waypoints: Sequence[tuple[float, float]], frame_count: int = OUTPUT_FRAME_COUNT
) -> list[tuple[float, float]]:
    """Resample a polyline to constant arc-length speed, including both endpoints."""

    if frame_count < 2:
        raise ValueError("frame_count must be at least two")
    points = np.asarray(waypoints, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] != 2:
        raise ValueError("At least two 2D waypoints are required")
    if not np.isfinite(points).all():
        raise ValueError("Waypoints must be finite")
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    keep = np.concatenate(([True], segment_lengths > 0))
    points = points[keep]
    if len(points) == 1:
        return [tuple(float(item) for item in points[0])] * frame_count
    cumulative = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))))
    samples = np.linspace(0.0, float(cumulative[-1]), frame_count)
    result = np.column_stack(
        (np.interp(samples, cumulative, points[:, 0]), np.interp(samples, cumulative, points[:, 1]))
    )
    return [(float(x), float(y)) for x, y in result]


def _extent(ellipse: Ellipse) -> tuple[float, float]:
    theta = math.radians(ellipse.angle_degrees)
    a, b = ellipse.major_diameter / 2.0, ellipse.minor_diameter / 2.0
    return (
        math.sqrt((a * math.cos(theta)) ** 2 + (b * math.sin(theta)) ** 2),
        math.sqrt((a * math.sin(theta)) ** 2 + (b * math.cos(theta)) ** 2),
    )


def clip_ellipse_to_frame(
    ellipse: Ellipse, *, width: int = TARGET_WIDTH, height: int = TARGET_HEIGHT
) -> tuple[Ellipse, dict[str, object] | None]:
    """Keep the complete rotated ellipse on-canvas, shrinking only if it cannot fit."""

    before = ellipse
    extent_x, extent_y = _extent(ellipse)
    scale = min(1.0, width / (2.0 * extent_x), height / (2.0 * extent_y))
    if scale < 1.0:
        ellipse = replace(
            ellipse,
            major_diameter=ellipse.major_diameter * scale,
            minor_diameter=ellipse.minor_diameter * scale,
        )
        extent_x, extent_y = _extent(ellipse)
    center_x = min(width - extent_x, max(extent_x, ellipse.center_x))
    center_y = min(height - extent_y, max(extent_y, ellipse.center_y))
    ellipse = replace(ellipse, center_x=center_x, center_y=center_y)
    if ellipse == before:
        return ellipse, None
    return ellipse, {
        "requested": [before.center_x, before.center_y],
        "applied": [ellipse.center_x, ellipse.center_y],
        "diameter_scale": scale,
        "reason": "full_rotated_ellipse_extent_clamped_to_canvas",
    }


def _selected_records(
    metadata: Mapping[str, object], instance_id: str
) -> list[Mapping[str, object]]:
    records = []
    for frame in metadata["frames"]:
        matches = [row for row in frame["instances"] if row.get("key") == instance_id]
        if len(matches) != 1:
            raise ValueError(
                f"Selected instance {instance_id!r} must appear exactly once in every frame; "
                f"frame {frame['frame_index']} has {len(matches)}"
            )
        records.append(matches[0])
    return records


def edit_trajectory(
    metadata: Mapping[str, object],
    *,
    source_conditioning: Path,
    source_metadata: Path,
    instance_id: str,
    waypoints: Sequence[tuple[float, float]],
    mode: str = "replace",
    simplify_epsilon_pixels: float = 2.0,
    seed: int = 0,
) -> tuple[dict[str, object], dict[str, object]]:
    """Apply a deterministic 97-frame centroid edit and return metadata + manifest."""

    if mode not in {"replace", "offset"}:
        raise ValueError("mode must be 'replace' or 'offset'")
    raw = [
        _finite_point(value, field=f"waypoints[{index}]") for index, value in enumerate(waypoints)
    ]
    if len(raw) < 2:
        raise ValueError("At least two waypoints are required")
    records = _selected_records(metadata, instance_id)
    classes = {str(record.get("class_name")) for record in records}
    if len(classes) != 1:
        raise ValueError("Selected instance class must remain constant across the sequence")
    class_name = next(iter(classes))
    original_ellipses = [_ellipse(record) for record in records]
    if not point_inside_ellipse(raw[0], original_ellipses[0]):
        raise ValueError("The first waypoint must begin inside the selected frame-zero ellipse")

    original_centroids = [(item.center_x, item.center_y) for item in original_ellipses]
    # Preserve frame zero exactly.  The initial click identifies the ellipse; the
    # centroid, rather than the potentially off-centre click, anchors the path.
    anchored = [original_centroids[0], *raw[1:]]
    simplified = simplify_waypoints(anchored, simplify_epsilon_pixels)
    interpolated = interpolate_waypoints(simplified, OUTPUT_FRAME_COUNT)
    if mode == "replace":
        requested_centroids = interpolated
    else:
        origin = interpolated[0]
        requested_centroids = [
            (source[0] + point[0] - origin[0], source[1] + point[1] - origin[1])
            for source, point in zip(original_centroids, interpolated, strict=True)
        ]

    edited = copy.deepcopy(metadata)
    clipping: list[dict[str, object]] = []
    edited_centroids: list[tuple[float, float]] = []
    for frame_index, (frame, requested) in enumerate(
        zip(edited["frames"], requested_centroids, strict=True)
    ):
        record = next(row for row in frame["instances"] if row.get("key") == instance_id)
        before = _ellipse(record)
        moved = replace(before, center_x=requested[0], center_y=requested[1])
        moved, clip = clip_ellipse_to_frame(moved)
        record["ellipse"] = asdict(moved)
        edited_centroids.append((moved.center_x, moved.center_y))
        if clip is not None:
            clipping.append({"frame_index": frame_index, **clip})

    edited["kind"] = "paper_ellipse_only_geometric_conditioning_trajectory_edit"
    edited["source_metadata_path"] = str(source_metadata)
    edited["trajectory_edit_manifest"] = "trajectory-edit-manifest.json"
    manifest: dict[str, object] = {
        "schema_version": 1,
        "kind": "interactive_ellipse_trajectory_edit",
        "classification": "reproduction_assumption_exact_author_interpolation_undisclosed",
        "source_conditioning_path": str(source_conditioning),
        "source_metadata_path": str(source_metadata),
        "selected_instance_id": instance_id,
        "selected_class": class_name,
        "original_centroids": [list(point) for point in original_centroids],
        "raw_waypoints": [list(point) for point in raw],
        "anchored_waypoints": [list(point) for point in anchored],
        "simplified_waypoints": [list(point) for point in simplified],
        "interpolated_waypoints": [list(point) for point in interpolated],
        "edited_centroids": [list(point) for point in edited_centroids],
        "coordinate_convention": COORDINATE_CONVENTION,
        "interpolation_method": INTERPOLATION_METHOD,
        "mode": mode,
        "mode_semantics": (
            "replace: selected centroid follows interpolated path; frame zero is anchored to "
            "the original centroid"
            if mode == "replace"
            else "offset: interpolated displacement is added to each source-frame centroid; "
            "frame zero has zero displacement"
        ),
        "simplify_epsilon_pixels": simplify_epsilon_pixels,
        "clipping": clipping,
        "random_seed": seed,
        "randomness_used": False,
        "software": {"package": "or-video-reproduction", "version": __version__},
        "video_contract": {
            "width": TARGET_WIDTH,
            "height": TARGET_HEIGHT,
            "frames": OUTPUT_FRAME_COUNT,
            "fps": OUTPUT_FPS,
            "background": "black",
        },
    }
    return edited, manifest


def _instances_from_record(frame: Mapping[str, object]) -> list[RenderInstance]:
    instances = []
    for row in frame["instances"]:
        class_name = str(row.get("class_name"))
        if class_name not in PAPER_36_PALETTE:
            raise ValueError(f"Unknown paper-36 class in metadata: {class_name!r}")
        normalized = row.get("normalized_nearness")
        if normalized is None:
            raise ValueError("Instance is missing normalized_nearness")
        instances.append(
            RenderInstance(
                key=str(row.get("key")),
                class_name=class_name,
                ellipse=_ellipse(row),
                normalized_depth=float(normalized),
                raw_depth=(
                    None
                    if row.get("raw_relative_depth_mean") is None
                    else float(row["raw_relative_depth_mean"])
                ),
            )
        )
    return instances


def rendered_frames(metadata: Mapping[str, object]) -> Iterable[NDArray[np.uint8]]:
    smaller_is_nearer = metadata.get("depth_direction") == "smaller_is_nearer"
    for frame in metadata["frames"]:
        yield render_conditioning(
            _instances_from_record(frame),
            (TARGET_HEIGHT, TARGET_WIDTH),
            smaller_is_nearer=smaller_is_nearer,
        )


def resolve_ffmpeg(executable: str | None = None) -> str:
    candidate = executable or shutil.which("ffmpeg")
    if not candidate:
        raise RuntimeError("ffmpeg is required to render the conditioning MP4")
    return candidate


def render_edited_sequence(
    metadata: Mapping[str, object], output: Path, *, ffmpeg: str | None = None
) -> None:
    """Render a bit-stable 1024x768/24-fps/97-frame H.264 MP4."""

    ffmpeg = resolve_ffmpeg(ffmpeg)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.stem + ".tmp.mp4")
    command = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        f"{TARGET_WIDTH}x{TARGET_HEIGHT}",
        "-framerate",
        str(OUTPUT_FPS),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-threads",
        "1",
        "-fflags",
        "+bitexact",
        "-flags:v",
        "+bitexact",
        "-map_metadata",
        "-1",
        str(temporary),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    try:
        for frame in rendered_frames(metadata):
            process.stdin.write(frame.tobytes(order="C"))
        process.stdin.close()
        return_code = process.wait()
    except BaseException:
        try:
            process.stdin.close()
        except (BrokenPipeError, OSError):
            pass
        process.kill()
        process.wait()
        temporary.unlink(missing_ok=True)
        raise
    if return_code:
        temporary.unlink(missing_ok=True)
        raise subprocess.CalledProcessError(return_code, command)
    temporary.replace(output)


def save_contact_sheet(metadata: Mapping[str, object], output: Path) -> None:
    wanted = {0, 24, 48, 72, 96}
    images = [
        Image.fromarray(frame)
        for index, frame in enumerate(rendered_frames(metadata))
        if index in wanted
    ]
    sheet = Image.new("RGB", (TARGET_WIDTH * len(images), TARGET_HEIGHT))
    for index, image in enumerate(images):
        sheet.paste(image, (index * TARGET_WIDTH, 0))
    sheet.save(output)


def run_edit(
    *,
    metadata_path: Path,
    source_conditioning: Path,
    trajectory_path: Path,
    output_dir: Path,
    instance_id: str | None = None,
    click: tuple[float, float] | None = None,
    seed: int = 0,
    ffmpeg: str | None = None,
) -> dict[str, object]:
    metadata = load_sequence_metadata(metadata_path)
    trajectory = load_trajectory(trajectory_path)
    selected_id, _ = select_instance(metadata, instance_id=instance_id, click=click)
    edited, manifest = edit_trajectory(
        metadata,
        source_conditioning=source_conditioning,
        source_metadata=metadata_path,
        instance_id=selected_id,
        waypoints=trajectory["waypoints"],
        mode=str(trajectory["mode"]),
        simplify_epsilon_pixels=float(trajectory["simplify_epsilon_pixels"]),
        seed=seed,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    render_edited_sequence(edited, output_dir / "conditioning-edited.mp4", ffmpeg=ffmpeg)
    save_contact_sheet(edited, output_dir / "conditioning-edited-contact-sheet.png")
    (output_dir / "metadata.json").write_text(
        json.dumps(edited, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest["edited_conditioning_path"] = str(output_dir / "conditioning-edited.mp4")
    manifest["edited_metadata_path"] = str(output_dir / "metadata.json")
    (output_dir / "trajectory-edit-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--source-conditioning", required=True, type=Path)
    parser.add_argument("--trajectory", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--instance-id")
    selection.add_argument("--click", nargs=2, type=float, metavar=("X", "Y"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ffmpeg", help="ffmpeg executable; defaults to PATH")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = run_edit(
        metadata_path=args.metadata,
        source_conditioning=args.source_conditioning,
        trajectory_path=args.trajectory,
        output_dir=args.output_dir,
        instance_id=args.instance_id,
        click=None if args.click is None else (args.click[0], args.click[1]),
        seed=args.seed,
        ffmpeg=args.ffmpeg,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def list_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="List stable instance IDs and ellipse geometry from conditioning metadata."
    )
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--frame", type=int, default=0)
    args = parser.parse_args(argv)
    metadata = load_sequence_metadata(args.metadata)
    print(json.dumps(list_instances(metadata, frame_index=args.frame), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
