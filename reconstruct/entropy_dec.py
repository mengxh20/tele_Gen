import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Deprecated entrypoint. End-to-end entropy decoding is now implicit in "
            "low_latents(v4) and recover/real_decoder flows."
        )
    )
    parser.add_argument("--input_path", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lines = [
        "Standalone entropy_dec.py has been removed from the main compression pipeline.",
        "There is no separate .bin entropy stage anymore.",
        "End-to-end entropy-coded bitstreams now live inside helios_low_latent_v4 payloads.",
        "Decode low latents with:",
        "  python reconstruct/real_decoder.py ...",
        "or call reconstruct.recover.decode_low_latents(...) directly.",
    ]
    if args.input_path is not None:
        lines.append(f"input_path={args.input_path.resolve()}")
    raise SystemExit("\n".join(lines))


if __name__ == "__main__":
    main()
