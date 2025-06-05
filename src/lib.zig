const std = @import("std");
const expect = std.testing.expect;

extern const _binary_trie_bin_start: opaque {};
extern const _binary_trie_root_bin_start: opaque {};
const vocab_trie: [*]const u8 = @ptrCast(&_binary_trie_bin_start);
const vocab_root_trie: [*]const usize = @alignCast(@ptrCast(&_binary_trie_root_bin_start));
const UNK: u16 = 101;

const TrieNode = struct {
    id: u16,
    size: u8,
    values: [*]const u8,
    offsets: [*]align(1) const usize,

    pub fn init(node: *const u8) TrieNode {
        const base = @intFromPtr(node);
        var id = @as(*align(1) const u16, @ptrFromInt(base)).*;
        if (id == -1) id = UNK;
        const size = @as(*const u8, @ptrFromInt(base + @sizeOf(u16))).*;
        const values = @as([*]const u8, @ptrFromInt(base + @sizeOf(u16) + @sizeOf(u8)));
        const offsets = @as([*]align(1) const usize, @ptrFromInt(base + @sizeOf(u16) + @sizeOf(u8) + size * @sizeOf(u8)));
        return .{
            .id = id,
            .size = size,
            .values = values,
            .offsets = offsets,
        };
    }

    pub fn debug(self: *TrieNode) void {
        std.debug.print("NODE (ID: {}, SIZE: {}, Values:", .{ self.id, self.size });
        for (0..self.size) |i| {
            std.debug.print(" {c}", .{self.values[i]});
        }
        std.debug.print(")\n", .{});
    }
};

test "1 word wordpiece encode" {
    var tokens: [64]u16 = undefined;
    const count = wordpeice_encode(vocab_root_trie, vocab_trie, "hello", &tokens);

    // this test is not fully correct
    // should have the start and end tokens!!!
    try expect(count == 1);
    try std.testing.expectEqual(7592, tokens[0]);
}

test "multi word wordpiece encode" {
    var tokens: [64]u16 = undefined;
    const count = wordpeice_encode(vocab_root_trie, vocab_trie, "embedding request from high script", &tokens);

    // this test is not fully correct
    // should have the start and end tokens!!!
    try expect(count == 7);
    try std.testing.expectEqual(7861, tokens[0]);
    try std.testing.expectEqual(8270, tokens[1]);
    try std.testing.expectEqual(4667, tokens[2]);
    try std.testing.expectEqual(5227, tokens[3]);
    try std.testing.expectEqual(2013, tokens[4]);
    try std.testing.expectEqual(2152, tokens[5]);
    try std.testing.expectEqual(5896, tokens[6]);
}

// TODO: handle utf-8 more gracefully
// TODO: add start and end tokens!
fn wordpeice_encode(req: EmbeddingRequestView) void {
    std.debug.print("Generating token ids for '{s}'\n", .{req.text[0..req.size.*]});
    var i: usize = 0;
    var root_offset: usize = 0;
    var token_count: usize = 0;
    top: while (i < req.size.*) {
        const c = req.text[i];
        switch (c) {
            // skip over whitespace
            // TODO: Think does `root_offset` need to be reset back to 0 here?
            ' ', '\t', '\n', '\r' => i += 1,
            // handle stuff for other characters
            else => {
                const offset = vocab_root_trie[c + root_offset];
                // need to check if this logic is needed
                // break out if root char invalid
                if (offset == -1) {
                    req.tokens[token_count] = UNK;
                    token_count += 1;
                    i += 1;
                    continue;
                }
                var found_tokens: [64]u16 = undefined;
                var found_token_idx: u8 = 1;
                var node = TrieNode.init(&vocab_trie[offset]);
                found_tokens[0] = node.id;
                inner: while (i + found_token_idx < req.size.*) {
                    const b = req.text[i + found_token_idx];

                    // TODO: we also need to handle whitespace
                    // that should start a new word!!!!

                    switch (b) {
                        ' ', '\n', '\t', '\r' => {
                            // NOTE: This code looks very simular to the code at the end of
                            // the prefix matching code
                            // but there is a small difference that we know that the next character
                            // is a white space so `i` may be advanced an extra spot since whitespace would
                            // normally just be skipped

                            for (0..found_token_idx) |j| {
                                const idx = found_token_idx - j - 1;
                                if (found_tokens[idx] != UNK) {
                                    req.tokens[token_count] = found_tokens[idx];
                                    token_count += 1;

                                    // this case means we had a valid word then a whitespace
                                    if (j == 0) {
                                        i += found_token_idx + 1;
                                        root_offset = 0;
                                    }
                                    // only a sub word matched
                                    else {
                                        i += idx;
                                        root_offset = 256;
                                    }

                                    continue :top;
                                }
                            }

                            // only happens if all found tokens are UNK
                            // advance i by 1 or 2 depending on if found tokens is 1
                            // conditionally set root_offset
                            req.tokens[token_count] = UNK;
                            token_count += 1;
                            // only 1 char then whitespace
                            // we can advance 2 (unknown char + the space)
                            if (found_token_idx == 1) {
                                i += 2;
                                root_offset = 0;
                            }
                            // n chars then whitespace
                            else {
                                i += 1;
                                root_offset = 256;
                            }

                            continue :top;
                        },
                        else => {},
                    }

                    for (0..node.size) |j| {
                        if (b == node.values[j]) {
                            node = TrieNode.init(&vocab_trie[node.offsets[j]]);
                            found_tokens[found_token_idx] = node.id;
                            found_token_idx += 1;
                            continue :inner;
                        }
                    }

                    // only will be here if no value is found
                    // we need to push the last found token
                    // then advance `i` by that much
                    // with a "##" prefix as root
                    for (0..found_token_idx) |j| {
                        const idx = found_token_idx - j - 1;
                        if (found_tokens[idx] != UNK) {
                            req.tokens[token_count] = found_tokens[idx];
                            token_count += 1;
                            i += idx + 1;
                            // TODO: do we need to set the ## conditionaly here?
                            root_offset = 256;
                            continue :top;
                        }
                    }

                    // will only be here if all found tokens are UNK
                    // only advance "i" by 1 and retry from the start
                    // with a "##" prefix as root
                    req.tokens[token_count] = UNK;
                    token_count += 1;
                    i += 1;
                    // TODO: do we need to set the ## conditionaly here?
                    root_offset = 256;
                    continue :top;
                }

                // if we are here it means that we have exhausted the full text
                for (0..found_token_idx) |j| {
                    const idx = found_token_idx - j - 1;
                    if (found_tokens[idx] != UNK) {
                        req.tokens[token_count] = found_tokens[idx];
                        token_count += 1;
                        if (j == 0) {
                            break :top;
                        }
                        i += idx;
                        // TODO: do we need to set the ## conditionaly here?
                        root_offset = 256;
                        continue :top;
                    }
                }

                // no matches but we hit the end of the string
                req.tokens[token_count] = UNK;
                token_count += 1;
                break :top;
            },
        }
    }

    std.debug.print("Encoded tokens: ", .{});
    for (0..token_count) |a| {
        std.debug.print("{} ", .{req.tokens[a]});
    }
    std.debug.print("\n", .{});

    req.is_done_tokenizing.* = true;
    req.tokens_count.* = 1; // really token chunk count
    req.pipeline().embedding_queue.push(.{ .offset = req.offset.*, .index = 0, .size = token_count, .batch = 1 }) catch {
        std.debug.print("FAILED TO enqueue embedding work\n", .{});
    };
}

fn RingQueue(comptime T: type, comptime N: u16) type {
    const M = N + 1;

    return struct {
        const Self = @This();

        buffer: [M]T,
        head: u16,
        tail: u16,

        pub fn init() Self {
            return .{ .head = 0, .tail = 0, .buffer = undefined };
        }

        pub fn is_empty(self: *Self) bool {
            return self.head == self.tail;
        }

        pub fn is_full(self: *Self) bool {
            return self.head == (self.tail + 1) % M;
        }

        pub fn peek(self: *Self) ?T {
            if (self.is_empty()) return null;
            return self.buffer[self.head];
        }

        pub fn reserve(self: *Self) !u16 {
            if (self.is_full()) return error.full;
            const idx = self.tail;
            self.tail = (self.tail + 1) % M;
            return idx;
        }

        pub fn insert(self: *Self, idx: u16, val: T) void {
            self.buffer[idx] = val;
        }

        pub fn discard(self: *Self) !void {
            if (self.is_empty()) return error.empty;
            self.head = (self.head + 1) % M;
        }

        pub fn push(self: *Self, val: T) !void {
            if (self.is_full()) return error.full;
            self.buffer[self.tail] = val;
            self.tail = (self.tail + 1) % M;
        }

        pub fn pop(self: *Self) !T {
            if (self.is_empty()) return error.empty;
            const val = self.buffer[self.head];
            self.head = (self.head + 1) % M;
            return val;
        }
    };
}

const GpuDevice = struct {
    stream: CudaStream,
    execution_context: TensorRTExecutionContext,
    input_ids_tensor: *anyopaque,
    attention_mask_tensor: *anyopaque,
    token_embeddings_tensor: *anyopaque,

    pub fn deinit(self: *GpuDevice) void {
        self.stream.deinit();
        self.execution_context.deinit();
    }
};

const TokenizationQueue = RingQueue(usize, 128);
const EmbeddingQueueItem = struct {
    offset: usize,
    index: usize,
    batch: usize,
    size: usize,
};
const EmbeddingQueue = RingQueue(EmbeddingQueueItem, 1024);

const InferencePipeline = struct {
    tokenization_queue: TokenizationQueue,
    embedding_queue: EmbeddingQueue,

    pub fn init() InferencePipeline {
        return .{
            .tokenization_queue = TokenizationQueue.init(),
            .embedding_queue = EmbeddingQueue.init(),
        };
    }

    pub fn enqueue(self: *InferencePipeline, offset: usize) !void {
        try self.tokenization_queue.push(offset);
    }

    pub fn tokenize(self: *InferencePipeline, base: usize, pool: *std.Thread.Pool) !void {
        const offset = try self.tokenization_queue.pop();
        pool.spawn(wordpeice_encode, .{EmbeddingRequestView.init(base + offset)}) catch |e| {
            std.debug.print("POOL SPAWN ERROR: {}\n", .{e});
        };
    }

    pub fn embed(self: *InferencePipeline, base: usize, device: *GpuDevice) !void {
        const item = try self.embedding_queue.pop();
        const ptr = base + item.offset;
        const req = EmbeddingRequestView.init(ptr);

        //var floats: [768]f32 = undefined;
        cudaMemcpyAsync(device.input_ids_tensor, req.tokens, item.batch * item.size * @sizeOf(u64), .h2d, device.stream.handle);
        device.execution_context.set_tensor_shape(@intCast(item.batch), @intCast(item.size));
        device.execution_context.set_tensor_address("input_ids", device.input_ids_tensor);
        device.execution_context.set_tensor_address("attention_mask", device.attention_mask_tensor); // TODO: this may or may not be needed
        device.execution_context.set_tensor_address("token_embeddings", device.token_embeddings_tensor);
        device.execution_context.enqueue(device.stream);
        cudaMemcpyAsync(req.embeddings, device.token_embeddings_tensor, item.batch * item.size * 768 * @sizeOf(f32), .d2h, device.stream.handle);
        _ = cudaLaunchHostFunc(device.stream.handle, cuda_embedding_callback, @ptrFromInt(ptr));
    }
};

export fn cuda_embedding_callback(data: ?*anyopaque) callconv(.C) void {
    const req = EmbeddingRequestView.init(@intFromPtr(data));
    for (0..10) |i| std.debug.print("{} ", .{req.embeddings[i]});
    std.debug.print("\n", .{});
}

const EmbeddingRequestView = struct {
    size: *usize,
    offset: *usize,
    pipeline_offset: *usize,
    tokens_count: *usize,
    embeddings_count: *usize,
    is_done_tokenizing: *bool,

    text: [*]u8,
    tokens: [*]u64,
    embeddings: [*]f32,

    fn ptr(comptime T: type, val: usize) T {
        return @as(T, @ptrFromInt(val));
    }

    fn field_offsets(text_size: usize) [10]usize {
        var offsets: [10]usize = undefined;
        var cum: usize = 0;

        offsets[0] = 0;
        cum += @sizeOf(usize);

        offsets[1] = cum;
        cum += @sizeOf(usize);

        offsets[2] = cum;
        cum += @sizeOf(usize);

        offsets[3] = cum;
        cum += @sizeOf(usize);

        offsets[4] = cum;
        cum += @sizeOf(usize);

        offsets[5] = cum;
        cum += @sizeOf(bool);

        offsets[6] = cum;
        cum = std.mem.alignForward(usize, cum + text_size, @sizeOf(u64));

        offsets[7] = cum;
        cum = std.mem.alignForward(usize, cum + @sizeOf(u16) * text_size, @sizeOf(f32));

        offsets[8] = cum;
        // this is a very conservative estimate of embedding space
        // TODO: improve this estimate once chunking strategies are implemented
        cum = std.mem.alignForward(usize, cum + 768 * text_size * @sizeOf(f32), 8);

        offsets[9] = cum;

        return offsets;
    }

    pub fn init(base: usize) EmbeddingRequestView {
        const size = @as(*usize, @ptrFromInt(base)).*;
        const offsets = EmbeddingRequestView.field_offsets(size);

        return .{
            .size = ptr(*usize, base + offsets[0]),
            .offset = ptr(*usize, base + offsets[1]),
            .pipeline_offset = ptr(*usize, base + offsets[2]),
            .tokens_count = ptr(*usize, base + offsets[3]),
            .embeddings_count = ptr(*usize, base + offsets[4]),
            .is_done_tokenizing = ptr(*bool, base + offsets[5]),
            .text = ptr([*]u8, base + offsets[6]),
            .tokens = ptr([*]u64, base + offsets[7]),
            .embeddings = ptr([*]f32, base + offsets[8]),
        };
    }

    pub fn pipeline(self: *const EmbeddingRequestView) *InferencePipeline {
        return @ptrFromInt(@intFromPtr(self.size) - self.pipeline_offset.*);
    }

    fn bytes(text_size: usize) usize {
        const offsets = EmbeddingRequestView.field_offsets(text_size);
        return offsets[9];
    }

    fn prepare_embedding_request(base: usize, offset: usize, pipeline_v: *InferencePipeline, text: []const u8) void {
        const view = EmbeddingRequestView.init(base + offset);
        view.size.* = text.len;
        for (0..text.len) |i| view.text[i] = text[i];
        view.is_done_tokenizing.* = false;
        view.pipeline_offset.* = base + offset - @intFromPtr(pipeline_v);
        view.tokens_count.* = 0;
        view.embeddings_count.* = 0;
        view.offset.* = offset;
    }
};

const SharedMemory = struct {
    head: usize,
    tail: usize,
    size: usize,

    pub fn init(size: usize) SharedMemory {
        return .{
            .head = 0,
            .tail = 8,
            .size = size,
        };
    }

    pub fn alloc(self: *SharedMemory, bytes: usize) !usize {
        if (self.tail == self.head) return error.full;
        if (self.tail < self.head) {
            if (self.head - self.tail >= bytes) {
                const t = self.tail;
                self.tail += bytes;
                return t;
            } else {
                return error.memory_full;
            }
        }

        if (self.size - self.tail >= bytes) {
            const t = self.tail;
            self.tail += bytes;
            return t;
        }

        if (self.head >= bytes) {
            self.tail = bytes;
            return 0;
        }

        return error.memory_full;
    }

    pub fn free(self: *SharedMemory, bytes: usize) void {
        if (self.head + bytes > self.size) {
            self.head = bytes;
        } else {
            self.head += bytes;
        }
    }
};

pub const InferenceEngine = struct {
    const GPU_MODELS = 1;
    const MAX_SEQUENCE_SIZE = 2048;
    const EMBEDDING_DIMENSION = 768;
    const MAX_BATCH_SIZE = 1;
    const MAX_INPUT_ID_BYTES = MAX_BATCH_SIZE * MAX_SEQUENCE_SIZE * @sizeOf(u64);
    const MAX_ATTENTION_MASK_BYTES = MAX_BATCH_SIZE * MAX_SEQUENCE_SIZE * @sizeOf(u64);
    const MAX_TOKEN_EMBEDDING_BYTES = MAX_BATCH_SIZE * MAX_SEQUENCE_SIZE * EMBEDDING_DIMENSION * @sizeOf(f32);
    const TENSOR_MEMORY_BYTES = MAX_ATTENTION_MASK_BYTES + GPU_MODELS * (MAX_INPUT_ID_BYTES + MAX_TOKEN_EMBEDDING_BYTES);

    shared_memory: SharedMemory,
    high_priority_pipeline: InferencePipeline,
    low_priority_pipeline: InferencePipeline,
    i: usize,

    // threading resources
    pool: std.Thread.Pool = undefined,

    // gpu resources
    model_runtime: TensorRTRuntime = undefined,
    model: TensorRTEngine = undefined,
    gpu_devices: [GPU_MODELS]GpuDevice = undefined,
    available_gpu_devices: RingQueue(u8, GPU_MODELS),

    pub fn init(size: usize) !InferenceEngine {
        return .{
            .high_priority_pipeline = InferencePipeline.init(),
            .low_priority_pipeline = InferencePipeline.init(),
            .shared_memory = SharedMemory.init(size - 0x10000),
            .i = 0,
            .available_gpu_devices = RingQueue(u8, GPU_MODELS).init(),
        };
    }

    pub fn start(self: *InferenceEngine, allocator: std.mem.Allocator) !void {
        try self.pool.init(.{ .allocator = allocator, .n_jobs = 4 });
        var gpu_memory: *anyopaque = undefined;
        _ = cudaMalloc(&gpu_memory, TENSOR_MEMORY_BYTES);
        var offset: usize = MAX_ATTENTION_MASK_BYTES;

        self.model_runtime = TensorRTRuntime.init();
        self.model = TensorRTEngine.init(&self.model_runtime);
        for (0..GPU_MODELS) |i| {
            self.gpu_devices[i].stream = CudaStream.init();
            self.gpu_devices[i].execution_context = TensorRTExecutionContext.init(&self.model);
            self.gpu_devices[i].attention_mask_tensor = gpu_memory;
            self.gpu_devices[i].input_ids_tensor = @ptrFromInt(@intFromPtr(gpu_memory) + offset);
            offset += MAX_INPUT_ID_BYTES;
            self.gpu_devices[i].token_embeddings_tensor = @ptrFromInt(@intFromPtr(gpu_memory) + offset);
            offset += MAX_TOKEN_EMBEDDING_BYTES;
            self.available_gpu_devices.push(@intCast(i)) catch unreachable;
        }
    }

    pub fn deinit(self: *InferenceEngine) void {
        self.pool.deinit();
        for (0..GPU_MODELS) |i| self.gpu_devices[i].deinit();
        self.model.deinit();
        self.model_runtime.deinit();

        // NOTE: this looks wrong but its correct
        _ = cudaFree(self.gpu_devices[0].attention_mask_tensor);
    }

    pub fn tick(self: *InferenceEngine) !void {
        if (self.i == 100000000) {
            std.debug.print("tick\n", .{});
            self.i = 0;
        }
        self.i += 1;

        const base = self.start_shared_memory_region();
        self.high_priority_pipeline.tokenize(base, &self.pool) catch {
            self.low_priority_pipeline.tokenize(base, &self.pool) catch {};
        };

        // TODO: need to add back the gpu once its done
        while (true) {
            const gpu_device_idx = self.available_gpu_devices.peek();
            if (gpu_device_idx != null) {
                const gpu_device = &self.gpu_devices[gpu_device_idx orelse unreachable];
                self.high_priority_pipeline.embed(base, gpu_device) catch {
                    self.low_priority_pipeline.embed(base, gpu_device) catch {
                        break;
                    };
                };
                self.available_gpu_devices.discard() catch unreachable;
            }
            break;
        }
    }

    fn start_shared_memory_region(self: *InferenceEngine) usize {
        return @intFromPtr(self) + @sizeOf(InferenceEngine);
    }

    fn enqueue_embedding_request(self: *InferenceEngine, pipeline: *InferencePipeline, text: []const u8) !usize {
        const size = EmbeddingRequestView.bytes(text.len);
        const offset = try self.shared_memory.alloc(size);
        EmbeddingRequestView.prepare_embedding_request(self.start_shared_memory_region(), offset, pipeline, text);
        try pipeline.enqueue(offset);
        return offset;
    }

    pub fn enqueue_high_priority_embedding_request(self: *InferenceEngine, text: []const u8) !usize {
        return self.enqueue_embedding_request(&self.high_priority_pipeline, text);
    }

    pub fn enqueue_low_priority_embedding_request(self: *InferenceEngine, text: []const u8) !usize {
        return self.enqueue_embedding_request(&self.low_priority_pipeline, text);
    }
};

pub fn create_memory_fd(size: usize) i32 {
    // switch to using huge tables
    const fd: i32 = @intCast(std.os.linux.memfd_create("mem", 0));
    _ = std.os.linux.ftruncate(fd, @intCast(size));
    return fd;
}

pub fn close_memory_fd(fd: i32) void {
    _ = std.os.linux.close(fd);
}

pub fn map_and_lock_fd(fd: i32, size: usize) [*]u8 {
    // todo switch to using huge tables
    // requires some system config
    const map_options = std.os.linux.MAP{ .TYPE = .SHARED, .LOCKED = true };
    const res = std.os.linux.mmap(null, size, std.os.linux.PROT.READ | std.os.linux.PROT.WRITE, map_options, fd, 0);
    const addr: [*]u8 = @ptrFromInt(res);

    return addr;
}

pub fn unlock_and_unmap_fd(addr: [*]const u8, size: usize) void {
    _ = std.os.linux.munmap(addr, size);
}

// cuda and custom kernel functions and wrappers
pub extern fn cudaMalloc(ptr: **anyopaque, size: usize) c_int;
pub extern fn cudaFree(data: *anyopaque) callconv(.C) u32;
pub const CudaMemcpyDirection = enum(c_int) {
    h2h,
    h2d,
    d2h,
    d2d,
};
pub extern fn cudaMemcpyAsync(dst: *anyopaque, src: *const anyopaque, size: usize, kind: CudaMemcpyDirection, stream: CudaStreamHandle) void;
pub extern fn cudaHostRegister(ptr: *anyopaque, size: usize, flags: u32) void;
pub extern fn cudaHostUnregister(ptr: *anyopaque) void;
pub extern fn cudaDeviceSynchronize() void;
pub extern fn cudaLaunchHostFunc(stream: CudaStreamHandle, func: ?*const fn (?*anyopaque) callconv(.C) void, data: ?*anyopaque) c_int;

const CudaStreamHandle = *anyopaque;
extern fn cudaStreamCreate(stream: *CudaStreamHandle) c_int;
extern fn cudaStreamDestroy(stream: CudaStreamHandle) c_int;
extern fn cudaStreamSynchronize(stream: CudaStreamHandle) void;
pub const CudaStream = struct {
    handle: CudaStreamHandle,

    pub fn init() CudaStream {
        var handle: CudaStreamHandle = undefined;
        _ = cudaStreamCreate(&handle);
        return .{ .handle = handle };
    }

    pub fn deinit(self: *CudaStream) void {
        _ = cudaStreamDestroy(self.handle);
    }

    pub fn sync(self: *CudaStream) void {
        cudaStreamSynchronize(self.handle);
    }

    pub fn record(self: *CudaStream, event: *CudaEvent) void {
        cudaEventRecord(event.handle, self.handle);
    }
};

const CudaEventHandle = *anyopaque;
extern fn cudaEventCreate(event: **anyopaque) void;
extern fn cudaEventRecord(event: *anyopaque, stream: CudaStreamHandle) void;
extern fn cudaEventSynchronize(event: *anyopaque) void;
pub const CudaEvent = struct {
    handle: CudaEventHandle,

    pub fn init() CudaEvent {
        const handle: CudaEventHandle = undefined;
        cudaEventCreate(&handle);
    }

    pub fn sync(self: *CudaEvent) void {
        cudaEventSynchronize(self.handle);
    }
};

// tensorRT functions
const TensorRTRuntimeHandle = *anyopaque;
extern fn create_runtime() *anyopaque;
extern fn destroy_runtime(rt: *anyopaque) void;
pub const TensorRTRuntime = struct {
    handle: TensorRTRuntimeHandle,

    pub fn init() TensorRTRuntime {
        const handle: TensorRTRuntimeHandle = create_runtime();
        return .{ .handle = handle };
    }

    pub fn deinit(self: *TensorRTRuntime) void {
        destroy_runtime(self.handle);
    }
};

const TensorRTEngineHandle = *anyopaque;
extern fn create_engine(rt: TensorRTRuntimeHandle, path: [*:0]const u8) TensorRTEngineHandle;
extern fn destroy_engine(eng: TensorRTEngineHandle) void;
extern fn engine_get_device_memory_size(eng: TensorRTEngineHandle) i64;
pub const TensorRTEngine = struct {
    handle: TensorRTEngineHandle,

    pub fn init(rt: *TensorRTRuntime) TensorRTEngine {
        return .{ .handle = create_engine(rt.handle, "model.engine") };
    }

    pub fn deinit(self: *TensorRTEngine) void {
        destroy_engine(self.handle);
    }

    pub fn get_device_memory_size(self: *TensorRTEngine) i64 {
        return engine_get_device_memory_size(self.handle);
    }
};

const TensorRTExecutionContextHandle = *anyopaque;
extern fn create_execution_context(eng: TensorRTEngineHandle) TensorRTExecutionContextHandle;
extern fn destroy_execution_context(ctx: TensorRTExecutionContextHandle) void;
extern fn execution_context_set_tensor_shape(ctx: TensorRTExecutionContextHandle, batch: i64, size: i64) void;
extern fn execution_context_set_device_memory(ctx: TensorRTExecutionContextHandle, ptr: *anyopaque) void;
extern fn execution_context_set_tensor_address(ctx: TensorRTExecutionContextHandle, name: [*c]const u8, ptr: *anyopaque) void;
extern fn execution_context_enqueue(ctx: TensorRTExecutionContextHandle, stream: CudaStreamHandle) void;
pub const TensorRTExecutionContext = struct {
    handle: TensorRTExecutionContextHandle,

    pub fn init(eng: *TensorRTEngine) TensorRTExecutionContext {
        return .{ .handle = create_execution_context(eng.handle) };
    }

    pub fn deinit(self: *TensorRTExecutionContext) void {
        destroy_execution_context(self.handle);
    }

    pub fn set_tensor_shape(self: *TensorRTExecutionContext, batch: i64, size: i64) void {
        execution_context_set_tensor_shape(self.handle, batch, size);
    }

    pub fn set_device_memory(self: *TensorRTExecutionContext, ptr: *anyopaque) void {
        execution_context_set_device_memory(self.handle, ptr);
    }

    pub fn set_tensor_address(self: *TensorRTExecutionContext, name: [*c]const u8, ptr: *anyopaque) void {
        execution_context_set_tensor_address(self.handle, name, ptr);
    }

    pub fn enqueue(self: *TensorRTExecutionContext, stream: CudaStream) void {
        execution_context_enqueue(self.handle, stream.handle);
    }
};

// custom kernels
pub extern fn upcast_uint16_to_int64() void;
