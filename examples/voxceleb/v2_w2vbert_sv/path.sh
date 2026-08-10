export PATH=$PWD:$PATH

# NOTE(kan-bayashi): Use UTF-8 in Python to avoid UnicodeDecodeError when LC_ALL=C
export PYTHONIOENCODING=UTF-8
# .deps holds optional pip packages (silero-vad/fire/onnxruntime/...) installed
# with: pip install --no-deps --target ../../../.deps <pkg>
export PYTHONPATH=../../../:../../../.deps:${PYTHONPATH:-}
