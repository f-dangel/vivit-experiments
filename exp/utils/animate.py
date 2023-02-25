"""Utility functions to create animations."""

from PIL import Image


def create_animation(
    frame_paths, savepath, format="GIF", duration=200, loop=0, verbose=True
):
    """Create animation from frames ans save to specified path."""
    if verbose:
        print("[exp] Creating animation from")
        for f_idx, f in enumerate(frame_paths):
            print(f"    Frame {f_idx:04d}: {f}")

    frame, *frames = [Image.open(f) for f in frame_paths]

    if verbose:
        print(f"[exp] Saving {format} animation to {savepath}")

    frame.save(
        fp=savepath,
        format=format,
        append_images=frames,
        save_all=True,
        duration=duration,
        loop=loop,
    )
