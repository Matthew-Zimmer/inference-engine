#include <vector>
#include <fstream>
#include <iostream>
#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>


__global__ void upcast_uint16_to_int64_kernel() {

}

extern "C" void upcast_uint16_to_int64(cudaStream_t stream) {
	upcast_uint16_to_int64_kernel<<<1, 1, 0, stream>>>();
}


class MyLogger : public nvinfer1::ILogger {
        void log(nvinfer1::ILogger::Severity severity, nvinfer1::AsciiChar const* msg) noexcept override {
                std::cout << msg << std::endl;
        }
} logger;

std::vector<char> read_engine_file(char* engine_file) {
    std::ifstream file(engine_file, std::ios::binary);

    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);

    std::vector<char> buffer(size);
    file.read(buffer.data(), size);
    return buffer;
}

extern "C" nvinfer1::IRuntime* create_runtime() {
	return nvinfer1::createInferRuntime(logger);
}

extern "C" void destroy_runtime(nvinfer1::IRuntime* rt) {
	delete rt;
}

extern "C" nvinfer1::ICudaEngine* create_engine(nvinfer1::IRuntime* rt, char* path) {
	auto vec = read_engine_file(path);
	return rt->deserializeCudaEngine(vec.data(), vec.size());
}

extern "C" void destroy_engine(nvinfer1::ICudaEngine* eng) {
	delete eng;
}

extern "C" nvinfer1::IExecutionContext* create_execution_context(nvinfer1::ICudaEngine* eng) {
	return eng->createExecutionContext();
}

extern "C" void destory_execution_context(nvinfer1::IExecutionContext* ctx) {
	delete ctx;
}

extern "C" void set_tensor_shape(nvinfer1::IExecutionContext* ctx, int32_t batch, int32_t size) {
	auto dims = nvinfer1::Dims64();
	dims.nbDims = 2;
	dims.d[0] = batch;
	dims.d[1] = size;
	ctx->setInputShape("input_ids", dims);
	ctx->setInputShape("attention_mask", dims);
}

extern "C" bool enqueue(nvinfer1::IExecutionContext* ctx, cudaStream_t stream) {
        return ctx->enqueueV3(stream);
}

extern "C" void set_device_memory(nvinfer1::IExecutionContext* ctx, void* address, int64_t size) {
	ctx->setDeviceMemoryV2(address, size);
}

extern "C" int64_t get_device_memory_size(nvinfer1::ICudaEngine* eng, int32_t profile) {
	return eng->getDeviceMemorySizeForProfileV2(profile);
}

