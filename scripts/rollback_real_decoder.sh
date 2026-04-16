#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

rm -f reconstruct/real_decoder.py

python - <<'PY'
from pathlib import Path


def replace_exact(path_str: str, old: str, new: str) -> None:
    path = Path(path_str)
    content = path.read_text(encoding="utf-8")
    if old not in content:
        return
    path.write_text(content.replace(old, new, 1), encoding="utf-8")


replace_exact(
    "scripts/inference/infer_recover_ddp.sh",
    """# 推理完之后自动走接收端链路：
# low_latents -> recover network -> recover_latents -> video
python reconstruct/real_decoder.py \\
    -R "${SAVEDIR}/recover_outputs" \\
    --checkpoint_dir "${SAVEDIR}/checkpoints"
""",
    """# 推理完之后自动进行解码恢复视频
python reconstruct/decoder.py -R ${SAVEDIR}/recover_outputs
""",
)

replace_exact(
    "reconstruct/AGENTS.md",
    """- `real_decoder.py`：接收端解码入口，优先读取 `low_latents`，再在内部调用 recover 网络恢复为 `recover_latents` 后解码成视频
""",
    "",
)

replace_exact(
    "reconstruct/AGENTS.md",
    """- 当需要更贴近真实接收端链路时，优先使用 `real_decoder.py` 走 `low_latents -> recover -> video` 路线
""",
    "",
)
PY

echo "Rolled back real_decoder changes."
