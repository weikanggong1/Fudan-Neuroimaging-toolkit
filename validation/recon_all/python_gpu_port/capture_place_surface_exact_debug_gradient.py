"""Read exact installed printf argument registers for a bounded pial gradient probe."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior-debug-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--license", type=Path, required=True)
    args = parser.parse_args()
    previous = json.loads(args.prior_debug_report.read_text())
    if previous["first_pass_iteration_limit"] != 1 or previous["debug_vertex"] != 80852:
        raise ValueError("expected the frozen first-step v80852 installed probe")
    command = previous["installed_command_argv"].copy()
    surf = args.out / "surf"
    surf.mkdir(parents=True, exist_ok=True)
    previous_input = Path(command[command.index("--i") + 1])
    copied_input = surf / previous_input.name
    shutil.copy2(previous_input, copied_input)
    command[command.index("--i") + 1] = str(copied_input)
    command[command.index("--o") + 1] = str(surf / "lh.pial.exact_debug")
    command[command.index("--target") + 1] = str(surf / "lh.pial.target")
    script = args.out / "exact_gradient.gdb"
    script.write_text('''set pagination off
set breakpoint pending on
break *0x416970
commands 1
 silent
 set {int}($rbx+0x514)=1
 continue
end
break *0x4176b0
commands 2
 silent
 set $rip=0x417776
 continue
end
break printf if $rdi == 0x7076b0
commands 3
 silent
 printf "EXACT_GRADIENT_VNO=%ld\\n", $rsi
 info registers xmm6 xmm7
 x/1gx $rsp+8
 continue
end
run
''')
    env = os.environ.copy()
    env.update(
        FREESURFER_HOME="/opt/reference/Freesurfer/8.2.0-1",
        SUBJECTS_DIR="/opt/reference/Freesurfer/8.2.0-1/subjects",
        FS_LICENSE=str(args.license),
    )
    log = args.out / "exact_gradient.log"
    with log.open("w") as output:
        subprocess.run(["gdb", "--batch", "-x", str(script), "--args", *command],
                       stdout=output, stderr=subprocess.STDOUT, env=env, check=True)
    lines = log.read_text().splitlines()
    captures = [index for index, line in enumerate(lines) if line.startswith("EXACT_GRADIENT_VNO=")]
    report = {
        "reference": "installed binary exact printf argument capture for selected first-step debug vertex; frozen n_averages=16/sigma2 input",
        "format_address": "0x7076b0",
        "method": "conditional printf breakpoint reads first eight promoted double arguments from XMM and ninth from stack; no binary mutation",
        "capture_count": len(captures),
        "capture_lines": [lines[index:index + 4] for index in captures],
        "mesh": str(surf / "lh.pial.exact_debug"),
        "native_argv": command,
    }
    (args.out / "exact_gradient_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"capture_count": report["capture_count"], "capture_lines": report["capture_lines"]}, indent=2))


if __name__ == "__main__":
    main()
