# To create an optimized inference engine

1. Pick the model from huggingface
2. python -m venv .venv
3. source .venv/bin/active
4. pip install -r requirements.txt
5. optimum-cli export onnx --model <model-name> --trust-remote-code onnx_model
6. trtrexec --onnx=onnx_model/model.onnx --saveEngine=model.engine --minShapes=NAME:DIM,NAME:DIM --optShapes=... --maxShapes=...
6a. You can find any shapes you need by running the command without any *Shapes args and see which tensors are auto set
6b. May need to mess with LD_LIBRARY_PATH to load different .so files

