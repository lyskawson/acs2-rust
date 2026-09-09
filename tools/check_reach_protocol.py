import json
from pathlib import Path
import subprocess
import tempfile


def main():
    repo = Path(__file__).resolve().parents[1]
    result = subprocess.run(["cargo", "build", "--release", "--bin", "mpx_reach", "--message-format=json"],
                            cwd=repo, check=True, text=True, stdout=subprocess.PIPE)
    libraries = {}
    for line in result.stdout.splitlines():
        message = json.loads(line)
        if message.get("reason") == "compiler-artifact":
            for filename in message["filenames"]:
                if filename.endswith(".rlib"):
                    libraries[message["target"]["name"]] = Path(filename)
    dependencies = libraries["acs2_bench"].parent
    with tempfile.TemporaryDirectory(prefix="acs2-reach-regressions-") as directory:
        binary = Path(directory) / "reach-regressions"
        command = ["rustc", "--edition=2021", "--test", "-O", "tools/reach_regressions.rs",
                   "-L", f"dependency={dependencies}", "-o", str(binary)]
        for crate in ("acs2_bench", "acs2_core", "acs2_envs", "libc"):
            library = libraries[crate]
            command.extend(["--extern", f"{crate}={library}"])
        subprocess.run(command, cwd=repo, check=True)
        subprocess.run([str(binary), "--test-threads=1"], check=True)


if __name__ == "__main__":
    main()
