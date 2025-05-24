all:
	./deps/cuda_nvcc/bin/nvcc -arch=sm_75 -Xcompiler -fPIC -c -I deps/cuda_cudart/include -I deps/TensorRT-10.10.0.31/include/ -L deps/cuda_cudart/lib64 src/model.cu -o gpu.o

