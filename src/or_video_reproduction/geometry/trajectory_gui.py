"""Interactive OpenCV/Pygame/Tk trajectory editor for ellipse conditioning."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .trajectory import (
    _ellipse,
    edit_trajectory,
    load_sequence_metadata,
    render_edited_sequence,
    rendered_frames,
    save_contact_sheet,
    select_instance,
)


def _choose_paths(
    metadata: Path | None, source: Path | None, output: Path | None
) -> tuple[Path, Path, Path]:
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    try:
        if metadata is None:
            value = filedialog.askopenfilename(
                title="Choose ellipse metadata.json", filetypes=[("JSON", "*.json")]
            )
            if not value:
                raise SystemExit("No metadata selected")
            metadata = Path(value)
        if source is None:
            value = filedialog.askopenfilename(
                title="Choose original conditioning MP4", filetypes=[("MP4", "*.mp4")]
            )
            if not value:
                raise SystemExit("No conditioning video selected")
            source = Path(value)
        if output is None:
            value = filedialog.askdirectory(title="Choose trajectory-edit output directory")
            if not value:
                raise SystemExit("No output directory selected")
            output = Path(value)
    finally:
        root.destroy()
    return metadata, source, output


def _save_result(
    *,
    metadata: dict[str, object],
    metadata_path: Path,
    source: Path,
    output: Path,
    instance_id: str,
    waypoints: list[tuple[float, float]],
    mode: str,
    epsilon: float,
    seed: int,
    ffmpeg: str | None,
) -> dict[str, object]:
    edited, manifest = edit_trajectory(
        metadata,
        source_conditioning=source,
        source_metadata=metadata_path,
        instance_id=instance_id,
        waypoints=waypoints,
        mode=mode,
        simplify_epsilon_pixels=epsilon,
        seed=seed,
    )
    output.mkdir(parents=True, exist_ok=True)
    video = output / "conditioning-edited.mp4"
    render_edited_sequence(edited, video, ffmpeg=ffmpeg)
    save_contact_sheet(edited, output / "conditioning-edited-contact-sheet.png")
    metadata_output = output / "metadata.json"
    metadata_output.write_text(
        json.dumps(edited, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest["edited_conditioning_path"] = str(video)
    manifest["edited_metadata_path"] = str(metadata_output)
    manifest_output = output / "trajectory-edit-manifest.json"
    manifest_output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def run_gui(
    *,
    metadata_path: Path | None = None,
    source_conditioning: Path | None = None,
    output_dir: Path | None = None,
    instance_id: str | None = None,
    mode: str = "replace",
    simplify_epsilon_pixels: float = 2.0,
    seed: int = 0,
    ffmpeg: str | None = None,
) -> dict[str, object] | None:
    """Open the interactive editor; Enter renders, R resets, Escape cancels."""

    metadata_path, source_conditioning, output_dir = _choose_paths(
        metadata_path, source_conditioning, output_dir
    )
    try:
        import cv2
        import numpy as np
        import pygame
    except ImportError as error:
        raise RuntimeError(
            "The interactive GUI needs the optional 'gui' dependencies: "
            "opencv-python, pygame, and Tk support"
        ) from error

    metadata = load_sequence_metadata(metadata_path)
    first_rgb = next(iter(rendered_frames(metadata)))
    first_bgr = cv2.cvtColor(first_rgb, cv2.COLOR_RGB2BGR)
    selected: str | None = None
    selected_class: str | None = None
    if instance_id is not None:
        selected, selected_class = select_instance(metadata, instance_id=instance_id)

    pygame.init()
    pygame.display.set_caption("Ellipse trajectory editor")
    screen = pygame.display.set_mode((1024, 768))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("arial", 21)
    small = pygame.font.SysFont("arial", 16)
    waypoints: list[tuple[float, float]] = []
    drawing = False
    message = "Click an ellipse, then drag a path. Enter renders; R resets; Esc cancels."

    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return None
                    if event.key == pygame.K_r:
                        waypoints = []
                        drawing = False
                        message = "Path reset. Click the selected ellipse and drag again."
                    if event.key == pygame.K_RETURN:
                        if selected is None or len(waypoints) < 2:
                            message = "Select an ellipse and draw at least two points first."
                        else:
                            message = "Rendering 97 frames…"
                            pygame.display.flip()
                            result = _save_result(
                                metadata=metadata,
                                metadata_path=metadata_path,
                                source=source_conditioning,
                                output=output_dir,
                                instance_id=selected,
                                waypoints=waypoints,
                                mode=mode,
                                epsilon=simplify_epsilon_pixels,
                                seed=seed,
                                ffmpeg=ffmpeg,
                            )
                            return result
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    point = float(event.pos[0]), float(event.pos[1])
                    if selected is None:
                        try:
                            selected, selected_class = select_instance(metadata, click=point)
                            waypoints = [point]
                            drawing = True
                            message = f"Selected {selected} ({selected_class}); keep dragging."
                        except ValueError as error:
                            message = str(error)
                    else:
                        row = next(
                            row
                            for row in metadata["frames"][0]["instances"]
                            if row.get("key") == selected
                        )
                        from .trajectory import point_inside_ellipse

                        if point_inside_ellipse(point, _ellipse(row)):
                            waypoints = [point]
                            drawing = True
                            message = f"Drawing path for {selected}."
                        else:
                            try:
                                selected, selected_class = select_instance(metadata, click=point)
                                waypoints = [point]
                                drawing = True
                                message = f"Selected {selected} ({selected_class}); keep dragging."
                            except ValueError:
                                message = "Begin the path inside an ellipse."
                if event.type == pygame.MOUSEMOTION and drawing and event.buttons[0]:
                    point = float(event.pos[0]), float(event.pos[1])
                    if (
                        not waypoints
                        or np.linalg.norm(np.asarray(point) - np.asarray(waypoints[-1])) >= 1.0
                    ):
                        waypoints.append(point)
                if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    drawing = False
                    if len(waypoints) >= 2:
                        message = f"Captured {len(waypoints)} points. Enter renders; R redraws."

            canvas = first_bgr.copy()
            if selected is not None:
                row = next(
                    row for row in metadata["frames"][0]["instances"] if row.get("key") == selected
                )
                ellipse = _ellipse(row)
                cv2.ellipse(
                    canvas,
                    (round(ellipse.center_x), round(ellipse.center_y)),
                    (round(ellipse.major_diameter / 2), round(ellipse.minor_diameter / 2)),
                    ellipse.angle_degrees,
                    0,
                    360,
                    (255, 255, 255),
                    3,
                    lineType=cv2.LINE_AA,
                )
            if len(waypoints) >= 2:
                cv2.polylines(
                    canvas,
                    [np.rint(np.asarray(waypoints)).astype(np.int32)],
                    False,
                    (255, 255, 255),
                    3,
                    lineType=cv2.LINE_AA,
                )
            surface = pygame.surfarray.make_surface(
                np.transpose(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), (1, 0, 2))
            )
            screen.blit(surface, (0, 0))
            overlay = pygame.Surface((1024, 64), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 190))
            screen.blit(overlay, (0, 0))
            screen.blit(font.render(message, True, (255, 255, 255)), (12, 8))
            detail = f"mode={mode}  epsilon={simplify_epsilon_pixels:g}px  output={output_dir}"
            screen.blit(small.render(detail, True, (200, 200, 200)), (12, 38))
            pygame.display.flip()
            clock.tick(60)
    finally:
        pygame.quit()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--source-conditioning", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--instance-id")
    parser.add_argument("--mode", choices=("replace", "offset"), default="replace")
    parser.add_argument("--simplify-epsilon-pixels", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ffmpeg")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_gui(
        metadata_path=args.metadata,
        source_conditioning=args.source_conditioning,
        output_dir=args.output_dir,
        instance_id=args.instance_id,
        mode=args.mode,
        simplify_epsilon_pixels=args.simplify_epsilon_pixels,
        seed=args.seed,
        ffmpeg=args.ffmpeg,
    )
    if result is not None:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
