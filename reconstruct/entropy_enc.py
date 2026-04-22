import argparse
import sys
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from reconstruct.latent_io import load_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Deprecated entrypoint. End-to-end entropy coding is now built into "
            "reconstruct/recover.py and low_latents(v4) generation."
        )
    )
    parser.add_argument("--input_path", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--metrics_dir", type=Path, default=None)
    return parser.parse_args()


def infer_payload_format(input_path: Optional[Path]) -> Optional[str]:
    if input_path is None:
        return None
    resolved = input_path.resolve()
    if not resolved.exists() or resolved.is_dir() or resolved.suffix.lower() != ".pt":
        return None
    try:
        payload = load_payload(resolved)
    except Exception:
        return None
    return str(payload.get("format_version", ""))


def build_message(input_path: Optional[Path], payload_format: Optional[str]) -> str:
    lines = [
        "Standalone entropy_enc.py has been removed from the main compression pipeline.",
        "The repo now uses end-to-end learned entropy coding inside reconstruct/recover.py.",
        "Current low_latents should be generated directly as helios_low_latent_v4 payloads.",
        "Use:",
        "  python reconstruct/recover.py infer ...",
        "Then decode with:",
        "  python reconstruct/real_decoder.py ...",
        "or call reconstruct.recover.decode_low_latents(...) directly.",
    ]
    if input_path is not None:
        lines.append(f"input_path={input_path.resolve()}")
    if payload_format:
        lines.append(f"detected_format_version={payload_format}")
        if payload_format == "helios_low_latent_v4":
            lines.append("This payload already contains end-to-end entropy-coded learned tail bitstreams.")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    payload_format = infer_payload_format(args.input_path)
    raise SystemExit(build_message(args.input_path, payload_format))


if __name__ == "__main__":
    main()
