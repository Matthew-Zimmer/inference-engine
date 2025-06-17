#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>

constexpr uint64_t embedding_dim = 768;

__global__ void upcast_uint16_to_int64_kernel(uint16_t const* __restrict__ input, uint64_t* __restrict__ output, uint64_t size) {
	uint64_t slot = blockIdx.x * blockDim.x + threadIdx.x;
	if (slot >= size) return;

	output[slot] = static_cast<uint64_t>(input[slot]);
}

extern "C" void upcast_uint16_to_int64(uint16_t const* __restrict__ input, uint64_t* __restrict__ output, uint64_t size, cudaStream_t stream) {
	int threads_per_block = 128;
	int blocks = (size + threads_per_block - 1) / threads_per_block;

	upcast_uint16_to_int64_kernel<<<blocks, threads_per_block, 0, stream>>>(input, output, size);
}

__global__ void average_token_embeddings_kernel(float const* __restrict__ input, float* __restrict__ output, uint64_t size) {
	uint64_t in_slot = blockIdx.x * size * embedding_dim + threadIdx.x ;
	uint64_t out_slot = threadIdx.x;

	for (uint64_t i = 0; i < size; ++i) {
		output[out_slot] += input[in_slot];
		in_slot += embedding_dim;
	}

	output[out_slot] /= static_cast<float>(size);
}

extern "C" void average_token_embeddings(float const* __restrict__ input, float* __restrict__ output, uint64_t batch, uint64_t size, cudaStream_t stream) {
	average_token_embeddings_kernel<<<batch, embedding_dim, 0, stream>>>(input, output, size);
}

class MyLogger : public nvinfer1::ILogger {
        void log(nvinfer1::ILogger::Severity severity, nvinfer1::AsciiChar const* msg) noexcept override {
		// printf("%s\n", msg);
	}
} logger;

struct EngineInfo {
	unsigned char* data;
	size_t size;
};

// TODO: handle error more gracefully
EngineInfo read_engine_file(char* engine_file) {
    FILE* file = fopen(engine_file, "rb");

    fseek(file, 0, SEEK_END);
    size_t size = ftell(file);
    rewind(file);

    unsigned char* data = (unsigned char*)malloc(size);
    fread(data, 1, size, file);

    fclose(file);
    EngineInfo info;
    info.data = data;
    info.size = size;
    return info;
}

extern "C" nvinfer1::IRuntime* create_runtime() {
	return nvinfer1::createInferRuntime(logger);
}

extern "C" void destroy_runtime(nvinfer1::IRuntime* rt) {
	delete rt;
}

extern "C" nvinfer1::ICudaEngine* create_engine(nvinfer1::IRuntime* rt, char* path) {
	EngineInfo info = read_engine_file(path);
	nvinfer1::ICudaEngine* eng = rt->deserializeCudaEngine(info.data, info.size);
	free(info.data);
	return eng;
}

extern "C" void destroy_engine(nvinfer1::ICudaEngine* eng) {
	delete eng;
}

extern "C" nvinfer1::IExecutionContext* create_execution_context(nvinfer1::ICudaEngine* eng) {
	return eng->createExecutionContext();
}

extern "C" void destroy_execution_context(nvinfer1::IExecutionContext* ctx) {
	delete ctx;
}

extern "C" void execution_context_set_tensor_shape(nvinfer1::IExecutionContext* ctx, int64_t batch, int64_t size) {
	auto dims = nvinfer1::Dims64();
	dims.nbDims = 2;
	dims.d[0] = batch;
	dims.d[1] = size;
	ctx->setInputShape("input_ids", dims);
	ctx->setInputShape("attention_mask", dims);
}

extern "C" void execution_context_set_tensor_address(nvinfer1::IExecutionContext* ctx, char const* name, void* ptr) {
	ctx->setTensorAddress(name, ptr);
}

extern "C" bool execution_context_enqueue(nvinfer1::IExecutionContext* ctx, cudaStream_t stream) {
        return ctx->enqueueV3(stream);
}

extern "C" void execution_context_set_device_memory(nvinfer1::IExecutionContext* ctx, void* address, int64_t size) {
	ctx->setDeviceMemoryV2(address, size);
}

extern "C" int64_t execution_context_get_device_memory_size(nvinfer1::ICudaEngine* eng, int32_t profile) {
	return eng->getDeviceMemorySizeForProfileV2(profile);
}

