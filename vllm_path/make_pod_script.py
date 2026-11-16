"""pod_template.sh + gpu_measure.py -> pod_run.sh (한 파일로 붙여넣기용)"""
import pathlib
d = pathlib.Path(__file__).parent
t = (d / "pod_template.sh").read_text(encoding="utf-8")
m = (d / "gpu_measure.py").read_text(encoding="utf-8")
assert "PYEOF" not in m
out = t.replace("@@GPU_MEASURE@@", m.rstrip("\n")).replace("\r\n", "\n")
(d / "pod_run.sh").write_text(out, encoding="utf-8", newline="\n")
print("pod_run.sh", len(out.splitlines()), "줄")
