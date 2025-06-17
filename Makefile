all:
	./deps/cuda_nvcc/bin/nvcc -O3 -arch=sm_75 -Xcompiler -fPIC -c -I deps/cuda_cudart/include -I deps/TensorRT-10.10.0.31/include/ -L deps/cuda_cudart/lib64 src/model.cu -o gpu.o

trie:
	ld -r -b binary -o trie.o trie.bin 
	ld -r -b binary -o trie_root.o trie_root.bin 
