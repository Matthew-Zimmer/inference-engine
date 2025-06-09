const std = @import("std");
const expect = std.testing.expect;

extern const _binary_trie_bin_start: opaque {};
extern const _binary_trie_root_bin_start: opaque {};
const vocab_trie: [*]const u8 = @ptrCast(&_binary_trie_bin_start);
const vocab_root_trie: [*]const usize = @alignCast(@ptrCast(&_binary_trie_root_bin_start));
const UNK: u16 = 100;
const CLS: u16 = 101;
const SEP: u16 = 102;

pub const SharedMemoryAllocator = struct {
    const MAX_ALLOCATIONS = 1000;

    const FreeListItem = struct {
        offset: usize,
        size: usize,
        in_use: bool,
    };

    memory_head: usize,
    memory_tail: usize,
    memory_size: usize,
    allocations_head: usize,
    allocations_tail: usize,
    allocations: [MAX_ALLOCATIONS]FreeListItem = undefined,

    pub fn init(size: usize) SharedMemoryAllocator {
        return .{
            .memory_head = 0,
            .memory_tail = 0,
            .memory_size = size,
            .allocations_head = 0,
            .allocations_tail = 0,
        };
    }

    fn next(idx: usize) usize {
        return (idx + 1) % MAX_ALLOCATIONS;
    }

    fn allocate(self: *SharedMemoryAllocator, size: usize, next_tail: usize) usize {
        const offset = self.memory_tail;
        self.allocations[self.allocations_tail] = .{ .offset = self.memory_tail, .size = size, .in_use = true };
        self.allocations_tail = next_tail;
        self.memory_tail += size;
        return offset;
    }

    pub fn alloc(self: *SharedMemoryAllocator, size: usize) !usize {
        const next_tail = next(self.allocations_tail);
        if (self.allocations_head == next_tail) return error.no_more_allocations_allowed;

        if (self.memory_tail >= self.memory_head) {
            if (self.memory_size - self.memory_tail >= size) {
                return self.allocate(size, next_tail);
            }

            if (self.memory_head >= size) {
                self.memory_tail = 0;
                return self.allocate(size, next_tail);
            }
        } else {
            if (self.memory_head - self.memory_tail >= size) {
                return self.allocate(size, next_tail);
            }
        }

        return error.not_enough_shared_memory;
    }

    fn free_allocation(self: *SharedMemoryAllocator, size: usize) void {
        self.allocations_head = next(self.allocations_head);
        if (self.memory_head + size <= self.memory_size) {
            self.memory_head += size;
        } else {
            self.memory_head = size;
        }
    }

    pub fn free(self: *SharedMemoryAllocator, offset: usize) void {
        var all_prior_freed: bool = true;

        var i = self.allocations_head;
        while (i != self.allocations_tail) {
            const item = &self.allocations[i];
            if (item.*.offset == offset) {
                item.*.in_use = false;
                if (all_prior_freed) {
                    self.free_allocation(item.*.size);
                }
                break;
            } else if (item.*.in_use) {
                all_prior_freed = false;
            } else {
                self.free_allocation(item.*.size);
            }
            i = next(i);
        }
    }
};

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

    pub fn lookup_next(self: *TrieNode, c: u8) ?TrieNode {
        for (0..self.size) |j| {
            if (c == self.values[j]) {
                return TrieNode.init(&vocab_trie[self.offsets[j]]);
            }
        }
        return null;
    }
};

fn is_whitespace(c: u8) bool {
    return c == ' ' or c == '\n' or c == '\t' or c == '\r';
}

const ChunkedWordPeiceEncoder = struct {
    const LOOK_AHEAD_SIZE = 64;
    const LARGE_CHUNK_SIZE = 2048 - 1; // minus 1 for the end token
    const SMALL_CHUNK_SIZE = 1024 - 1; // minus 1 for the end token

    req: ChunkedEmbeddingRequestView,
    index: usize,
    current_page: usize,
    root_offset: usize,
    large_chunk_index: usize,
    large_chunk_start_page: usize,
    small_chunk_index: usize,
    small_chunk_start_page: usize,
    page_chunk_index: usize,
    page_chunk_start_page: usize,
    look_ahead_index: usize,
    token_look_ahead: [LOOK_AHEAD_SIZE]u16 = undefined,

    pub fn init(req: ChunkedEmbeddingRequestView) ChunkedWordPeiceEncoder {
        return .{
            .req = req,
            .index = 0,
            .current_page = 0,
            .root_offset = 0,
            .large_chunk_index = 0,
            .large_chunk_start_page = 0,
            .small_chunk_index = 0,
            .small_chunk_start_page = 0,
            .page_chunk_index = 0,
            .page_chunk_start_page = 0,
            .look_ahead_index = 0,
        };
    }

    fn peek(self: *ChunkedWordPeiceEncoder, offset: usize) u8 {
        const c = self.req.text[self.index + offset];
        return c;
    }

    fn advance(self: *ChunkedWordPeiceEncoder, stride: usize) void {
        self.index += stride;

        while (self.index >= self.req.page_offsets[self.current_page]) {
            self.current_page += 1;
        }
    }

    fn emit_token(self: *ChunkedWordPeiceEncoder, token: u16) void {
        // copy token to large token stream
        self.req.large_chunk_tokens[self.req.large_tokens_count.*] = token;
        self.req.large_tokens_count.* += 1;

        if (self.req.large_tokens_count.* % LARGE_CHUNK_SIZE == 0) {
            self.req.large_chunk_tokens[self.req.large_tokens_count.*] = SEP;
            self.req.large_tokens_count.* += 1;

            const embedding_idx = self.req.large_chunks_count.* * InferenceEngine.EMBEDDING_DIMENSION;
            self.req.large_chunks_count.* += 1;
            self.req.pipeline().embedding_queue.push(.{
                .offset = self.req.offset.*,
                .size = LARGE_CHUNK_SIZE,
                .batch = 1,
                .tokens = @ptrCast(&self.req.large_chunk_tokens[self.large_chunk_index]),
                .embeddings = @ptrCast(&self.req.large_chunk_embeddings[embedding_idx]),
            }) catch {
                // TODO: wtf to do here????
                // we failed to insert the large chunk token
            };

            self.large_chunk_index += LARGE_CHUNK_SIZE + 1;
            self.req.large_chunk_tokens[self.req.large_tokens_count.*] = CLS;

            self.req.large_tokens_count.* += 1;
            self.large_chunk_start_page = self.current_page;
        }

        // copy token to small token stream
        self.req.small_chunk_tokens[self.req.small_tokens_count.*] = token;
        self.req.small_tokens_count.* += 1;

        if (self.req.small_tokens_count.* % SMALL_CHUNK_SIZE == 0) {
            self.req.small_chunk_tokens[self.req.small_tokens_count.*] = SEP;
            self.req.small_tokens_count.* += 1;

            const embedding_idx = self.req.small_chunks_count.* * InferenceEngine.EMBEDDING_DIMENSION;
            self.req.small_chunks_count.* += 1;
            self.req.pipeline().embedding_queue.push(.{
                .offset = self.req.offset.*,
                .size = SMALL_CHUNK_SIZE,
                .batch = 1,
                .tokens = @ptrCast(&self.req.small_chunk_tokens[self.small_chunk_index]),
                .embeddings = @ptrCast(&self.req.small_chunk_embeddings[embedding_idx]),
            }) catch {
                // TODO: wtf to do here????
                // we failed to insert the small chunk token
            };

            self.small_chunk_index += SMALL_CHUNK_SIZE + 1;
            self.req.small_chunk_tokens[self.req.small_tokens_count.*] = CLS;

            self.req.small_tokens_count.* += 1;
            self.small_chunk_start_page = self.current_page;
        }

        // copy token to page token stream
        self.req.page_chunk_tokens[self.req.page_tokens_count.*] = token;
        self.req.page_tokens_count.* += 1;

        if (self.page_chunk_start_page != self.current_page) {
            self.req.page_chunk_tokens[self.req.page_tokens_count.*] = SEP;
            self.req.page_tokens_count.* += 1;

            const size = self.req.page_tokens_count.* - self.page_chunk_index;

            const embedding_idx = self.req.page_chunks_count.* * InferenceEngine.EMBEDDING_DIMENSION;
            self.req.page_chunks_count.* += 1;
            self.req.pipeline().embedding_queue.push(.{
                .offset = self.req.offset.*,
                .size = size,
                .batch = 1,
                .tokens = @ptrCast(&self.req.page_chunk_tokens[self.page_chunk_index]),
                .embeddings = @ptrCast(&self.req.page_chunk_embeddings[embedding_idx]),
            }) catch {
                // TODO: wtf to do here????
                // we failed to insert the small chunk token
            };

            self.req.page_chunk_tokens[self.req.page_tokens_count.*] = CLS;
            self.req.page_tokens_count.* += 1;

            self.page_chunk_index += size + 1;
            self.page_chunk_start_page = self.current_page;
        }
    }

    fn assign_root_offset(self: *ChunkedWordPeiceEncoder) void {
        if (self.index < self.req.size.*) {
            self.root_offset = if (is_whitespace(self.peek(0))) 0 else 256;
        }
        self.look_ahead_index = 0;
    }

    fn add_look_ahead(self: *ChunkedWordPeiceEncoder, token: u16) void {
        self.token_look_ahead[self.look_ahead_index] = token;
        self.look_ahead_index += 1;
    }

    fn commit_look_ahead(self: *ChunkedWordPeiceEncoder) void {
        for (0..self.look_ahead_index) |i| {
            const index = self.look_ahead_index - i - 1;
            const token = self.token_look_ahead[index];
            if (token != UNK) {
                self.advance(index + 1);
                self.emit_token(token);
                self.assign_root_offset();
                return;
            }
        }

        self.advance(1);
        self.emit_token(UNK);
        self.assign_root_offset();
    }

    pub fn encode(self: *ChunkedWordPeiceEncoder) void {
        // std.debug.print("Generating token ids for '{s}'\n", .{self.req.text[0..self.req.size.*]});
        const size = self.req.size.*;

        while (self.index < size) {
            inner: while (self.index < size) {
                const c = self.peek(0);
                if (is_whitespace(c)) {
                    self.advance(1);
                    continue :inner;
                }

                const offset = vocab_root_trie[c];
                if (offset == -1) {
                    self.advance(1);
                    self.emit_token(UNK);
                    continue :inner;
                }
                var node = TrieNode.init(&vocab_trie[vocab_root_trie[c + self.root_offset]]);
                self.add_look_ahead(node.id);
                const max_look_ahead = @min(LOOK_AHEAD_SIZE, size - self.index);
                for (1..max_look_ahead) |i| {
                    node = node.lookup_next(self.peek(i)) orelse {
                        self.commit_look_ahead();
                        continue :inner;
                    };
                    self.add_look_ahead(node.id);
                }
                self.commit_look_ahead();
            }

            // there exists some characters that have not been converted into tokens
            // NOTE: its not ganenteed that this will consume all characters
            // the look ahead buffer may no have a partial match even though
            // it fully walked the input stream
            // EXAMPLE: embed (assumming embedded is in the vocab but embed is not)
            // the this would not consume all characters and would need another go around
            // till either the look ahead is empty or it consumes all characters
            if (self.look_ahead_index > 0) {
                self.commit_look_ahead();
            }
        }

        // there exists a non complete chunk of large chunked tokens
        if (self.large_chunk_index != self.req.large_tokens_count.*) {
            self.req.large_chunk_tokens[self.req.large_tokens_count.*] = SEP;
            self.req.large_tokens_count.* += 1;

            const remainding_tokens = self.req.large_tokens_count.* - self.large_chunk_index;

            const embedding_idx = self.req.large_chunks_count.* * InferenceEngine.EMBEDDING_DIMENSION;
            self.req.large_chunks_count.* += 1;
            self.req.pipeline().embedding_queue.push(.{
                .offset = self.req.offset.*,
                .size = remainding_tokens,
                .batch = 1,
                .tokens = @ptrCast(&self.req.large_chunk_tokens[self.large_chunk_index]),
                .embeddings = @ptrCast(&self.req.large_chunk_embeddings[embedding_idx]),
            }) catch {
                // TODO: wtf to do here????
                // we failed to insert the large chunk token
            };
        }

        // there exists a non complete chunk of small chunked tokens
        if (self.small_chunk_index != self.req.small_tokens_count.*) {
            self.req.small_chunk_tokens[self.req.small_tokens_count.*] = SEP;
            self.req.small_tokens_count.* += 1;

            const remainding_tokens = self.req.small_tokens_count.* - self.small_chunk_index;

            const embedding_idx = self.req.small_chunks_count.* * InferenceEngine.EMBEDDING_DIMENSION;
            self.req.small_chunks_count.* += 1;
            self.req.pipeline().embedding_queue.push(.{
                .offset = self.req.offset.*,
                .size = remainding_tokens,
                .batch = 1,
                .tokens = @ptrCast(&self.req.small_chunk_tokens[self.small_chunk_index]),
                .embeddings = @ptrCast(&self.req.small_chunk_embeddings[embedding_idx]),
            }) catch {
                // TODO: wtf to do here????
                // we failed to insert the small chunk token
            };
        }

        // there exists a non complete chunk of page chunked tokens
        if (self.page_chunk_index != self.req.page_tokens_count.*) {
            self.req.page_chunk_tokens[self.req.page_tokens_count.*] = SEP;
            self.req.page_tokens_count.* += 1;

            const remainding_tokens = self.req.page_tokens_count.* - self.page_chunk_index;

            const embedding_idx = self.req.page_chunks_count.* * InferenceEngine.EMBEDDING_DIMENSION;
            self.req.page_chunks_count.* += 1;
            self.req.pipeline().embedding_queue.push(.{
                .offset = self.req.offset.*,
                .size = remainding_tokens,
                .batch = 1,
                .tokens = @ptrCast(&self.req.page_chunk_tokens[self.page_chunk_index]),
                .embeddings = @ptrCast(&self.req.page_chunk_embeddings[embedding_idx]),
            }) catch {
                // TODO: wtf to do here????
                // we failed to insert the large chunk token
            };
        }

        self.req.is_done_tokenizing.* = true;
    }
};

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
    skinny_input_ids_tensor: *anyopaque,
    fat_input_ids_tensor: *anyopaque,
    attention_mask_tensor: *anyopaque,
    token_embeddings_tensor: *anyopaque,

    pub fn deinit(self: *GpuDevice) void {
        self.stream.deinit();
        self.execution_context.deinit();
    }
};

const TokenizationQueue = RingQueue(usize, 128);
const ChunkType = enum {
    large,
    small,
    page,
};
const EmbeddingQueueItem = struct {
    offset: usize,
    batch: usize,
    size: usize,
    tokens: [*]u16,
    embeddings: [*]f32,
};
const EmbeddingQueue = RingQueue(EmbeddingQueueItem, 1024);
const NotifyQueue = RingQueue(usize, 128);
const EmbeddingCallbackData = struct {
    request_offset: usize,
    device_idx: u8,
};

const InferencePipeline = struct {
    tokenization_queue: TokenizationQueue,
    embedding_queue: EmbeddingQueue,
    notify_queue: NotifyQueue,
    active_requests: [InferenceEngine.GPU_MODELS]EmbeddingCallbackData = undefined,

    pub fn init() InferencePipeline {
        return .{
            .tokenization_queue = TokenizationQueue.init(),
            .embedding_queue = EmbeddingQueue.init(),
            .notify_queue = NotifyQueue.init(),
        };
    }

    pub fn enqueue(self: *InferencePipeline, offset: usize) !void {
        try self.tokenization_queue.push(offset);
    }

    pub fn tokenize(self: *InferencePipeline, base: usize, pool: *std.Thread.Pool) !void {
        const offset = try self.tokenization_queue.pop();
        pool.spawn(wordpeice_encode, .{ChunkedEmbeddingRequestView.view(base + offset)}) catch |e| {
            std.debug.print("POOL SPAWN ERROR: {}\n", .{e});
        };
    }

    pub fn embed(self: *InferencePipeline, base: usize, device: *GpuDevice, idx: u8) !void {
        const item = try self.embedding_queue.pop();
        const ptr = base + item.offset;
        self.active_requests[idx] = .{ .request_offset = ptr, .device_idx = idx };

        cudaMemcpyAsync(device.skinny_input_ids_tensor, item.tokens, item.batch * item.size * @sizeOf(u64), .h2d, device.stream.handle);
        upcast_uint16_to_int64(@alignCast(@ptrCast(device.skinny_input_ids_tensor)), @alignCast(@ptrCast(device.fat_input_ids_tensor)), item.batch * item.size, device.stream.handle);
        device.execution_context.set_tensor_shape(@intCast(item.batch), @intCast(item.size));
        device.execution_context.set_tensor_address("input_ids", device.fat_input_ids_tensor);
        device.execution_context.set_tensor_address("attention_mask", device.attention_mask_tensor); // TODO: this may or may not be needed
        device.execution_context.set_tensor_address("token_embeddings", device.token_embeddings_tensor);
        device.execution_context.enqueue(device.stream);
        average_token_embeddings(@alignCast(@ptrCast(device.token_embeddings_tensor)), item.batch, item.size, device.stream.handle);
        cudaMemcpyAsync(item.embeddings, device.token_embeddings_tensor, item.batch * 768 * @sizeOf(f32), .d2h, device.stream.handle);
        _ = cudaLaunchHostFunc(device.stream.handle, cuda_embedding_callback, &self.active_requests[idx]);
    }

    pub fn notify(self: *InferencePipeline, base: usize, eng: *InferenceEngine) !void {
        const item = try self.notify_queue.pop();
        const ptr = base + item;
        const req = ChunkedEmbeddingRequestView.view(ptr);

        var file = try std.fs.cwd().createFile("request_data.bin", .{});
        defer file.close();

        _ = try file.write(req.raw());

        const val: u64 = 1;
        _ = std.os.linux.write(req.event_fd.*, @ptrCast(&val), @sizeOf(u64));
        eng.events.push(req.event_fd.*) catch unreachable;
    }
};

fn wordpeice_encode(req: ChunkedEmbeddingRequestView) void {
    var encoder = ChunkedWordPeiceEncoder.init(req);
    encoder.encode();
}

export fn cuda_embedding_callback(data: ?*anyopaque) callconv(.C) void {
    const active_request: *EmbeddingCallbackData = @alignCast(@ptrCast(data));
    const req = ChunkedEmbeddingRequestView.view(active_request.request_offset);
    defer req.engine().available_gpu_devices.push(active_request.device_idx) catch unreachable;

    req.embeddings_count.* += 1;
    const token_chunks_count = req.large_chunks_count.* + req.small_chunks_count.* + req.page_chunks_count.*;

    if (req.is_done_tokenizing.* and token_chunks_count == req.embeddings_count.*) {
        req.pipeline().notify_queue.push(req.offset.*) catch |e| {
            // TODO: WTF to do here?
            std.debug.print("notify queue push err: {}\n", .{e});
        };
    }
}

fn number_of_groups(n: usize, size: usize) usize {
    return n / size + @intFromBool(n % size > 0);
}

pub const ChunkedEmbeddingRequestView = struct {
    pub const MODEL_DIM = 768;

    size: *usize,
    pages: *usize,
    offset: *usize,
    pipeline_offset: *usize,
    large_tokens_count: *usize,
    small_tokens_count: *usize,
    page_tokens_count: *usize,
    large_chunks_count: *usize,
    small_chunks_count: *usize,
    page_chunks_count: *usize,
    embeddings_count: *usize,
    event_fd: *i32,
    is_done_tokenizing: *bool,

    text: [*]u8,
    page_offsets: [*]u64,
    large_chunk_tokens: [*]u16,
    small_chunk_tokens: [*]u16,
    page_chunk_tokens: [*]u16,
    large_chunk_embeddings: [*]f32,
    small_chunk_embeddings: [*]f32,
    page_chunk_embeddings: [*]f32,

    fn ptr(comptime T: type, val: usize) T {
        return @as(T, @ptrFromInt(val));
    }

    fn field_offsets(text_size: usize, pages: usize) [22]usize {
        var offsets: [22]usize = undefined;
        var cum: usize = 0;

        const max_large_chunks = number_of_groups(text_size, ChunkedWordPeiceEncoder.LARGE_CHUNK_SIZE);
        const max_small_chunks = number_of_groups(text_size, ChunkedWordPeiceEncoder.SMALL_CHUNK_SIZE);
        // TODO: This assumes that 1 page will never have more then InferenceEngine.MAX_TOKENS which is most likely not true in general
        // we will need to add more chunks to this but its a lower bound
        const max_page_chunks = pages;

        // size
        offsets[0] = 0;
        cum += @sizeOf(usize);

        // pages
        offsets[1] = cum;
        cum += @sizeOf(usize);

        // offset
        offsets[2] = cum;
        cum += @sizeOf(usize);

        // pipeline_offset
        offsets[3] = cum;
        cum += @sizeOf(usize);

        // large_tokens_count
        offsets[4] = cum;
        cum += @sizeOf(usize);

        // small_tokens_count
        offsets[5] = cum;
        cum += @sizeOf(usize);

        // page_tokens_count
        offsets[6] = cum;
        cum += @sizeOf(usize);

        // large chunks count
        offsets[7] = cum;
        cum += @sizeOf(usize);

        // small chunks count
        offsets[8] = cum;
        cum += @sizeOf(usize);

        // page chunks count
        offsets[9] = cum;
        cum += @sizeOf(usize);

        // embeddings_count
        offsets[10] = cum;
        cum += @sizeOf(usize);

        // event fd
        offsets[11] = cum;
        cum += @sizeOf(i32);

        // is_done_tokenizing
        offsets[12] = cum;
        cum += @sizeOf(bool);

        // text
        offsets[13] = cum;
        cum = std.mem.alignForward(usize, cum + text_size, @sizeOf(u64));

        // page offsets
        offsets[14] = cum;
        cum = std.mem.alignForward(usize, cum + @sizeOf(u64) * pages, @sizeOf(u16));

        // large chunk tokens
        offsets[15] = cum;
        cum = std.mem.alignForward(usize, cum + @sizeOf(u16) * text_size + 2 * max_large_chunks, @sizeOf(u16));

        // small chunk tokens
        offsets[16] = cum;
        cum = std.mem.alignForward(usize, cum + @sizeOf(u16) * text_size + 2 * max_small_chunks, @sizeOf(u16));

        // page chunk tokens
        offsets[17] = cum;
        cum = std.mem.alignForward(usize, cum + @sizeOf(u16) * text_size + 2 * max_page_chunks, @sizeOf(f32));

        // large chunk embeddings
        offsets[18] = cum;
        cum = std.mem.alignForward(usize, cum + MODEL_DIM * max_large_chunks * @sizeOf(f32), @sizeOf(f32));

        // small chunk embeddings
        offsets[19] = cum;
        cum = std.mem.alignForward(usize, cum + MODEL_DIM * max_small_chunks * @sizeOf(f32), @sizeOf(f32));

        // page chunk embeddings
        offsets[20] = cum;
        cum = std.mem.alignForward(usize, cum + MODEL_DIM * max_page_chunks * @sizeOf(f32), 8);

        // total size
        offsets[21] = cum;

        return offsets;
    }

    pub fn view(base: usize) ChunkedEmbeddingRequestView {
        const size = @as(*usize, @ptrFromInt(base)).*;
        const pages = @as(*usize, @ptrFromInt(base + @sizeOf(usize))).*;

        return ChunkedEmbeddingRequestView.init(base, size, pages);
    }

    fn init(base: usize, size: usize, pages: usize) ChunkedEmbeddingRequestView {
        const offsets = ChunkedEmbeddingRequestView.field_offsets(size, pages);

        return .{
            .size = ptr(*usize, base + offsets[0]),
            .pages = ptr(*usize, base + offsets[1]),
            .offset = ptr(*usize, base + offsets[2]),
            .pipeline_offset = ptr(*usize, base + offsets[3]),
            .large_tokens_count = ptr(*usize, base + offsets[4]),
            .small_tokens_count = ptr(*usize, base + offsets[5]),
            .page_tokens_count = ptr(*usize, base + offsets[6]),
            .large_chunks_count = ptr(*usize, base + offsets[7]),
            .small_chunks_count = ptr(*usize, base + offsets[8]),
            .page_chunks_count = ptr(*usize, base + offsets[9]),
            .embeddings_count = ptr(*usize, base + offsets[10]),
            .event_fd = ptr(*i32, base + offsets[11]),
            .is_done_tokenizing = ptr(*bool, base + offsets[12]),
            .text = ptr([*]u8, base + offsets[13]),
            .page_offsets = ptr([*]u64, base + offsets[14]),
            .large_chunk_tokens = ptr([*]u16, base + offsets[15]),
            .small_chunk_tokens = ptr([*]u16, base + offsets[16]),
            .page_chunk_tokens = ptr([*]u16, base + offsets[17]),
            .large_chunk_embeddings = ptr([*]f32, base + offsets[18]),
            .small_chunk_embeddings = ptr([*]f32, base + offsets[19]),
            .page_chunk_embeddings = ptr([*]f32, base + offsets[20]),
        };
    }

    pub fn pipeline(self: *const ChunkedEmbeddingRequestView) *InferencePipeline {
        return @ptrFromInt(@intFromPtr(self.size) - self.pipeline_offset.*);
    }

    pub fn engine(self: *const ChunkedEmbeddingRequestView) *InferenceEngine {
        return @ptrFromInt(@intFromPtr(self.size) - self.offset.* - @sizeOf(InferenceEngine));
    }

    fn bytes(text_size: usize, pages: usize) usize {
        const offsets = ChunkedEmbeddingRequestView.field_offsets(text_size, pages);
        return offsets[21];
    }

    pub fn raw(self: *const ChunkedEmbeddingRequestView) []u8 {
        return @as([*]u8, @alignCast(@ptrCast(self.size)))[0..bytes(self.size.*, self.pages.*)];
    }

    fn init_data(base: usize, offset: usize, pipeline_v: *InferencePipeline, event: i32, text: []const u8, pages: []const u64) void {
        const req = ChunkedEmbeddingRequestView.init(base + offset, text.len, pages.len);

        req.size.* = text.len;
        req.pages.* = pages.len;
        for (0..text.len) |i| req.text[i] = text[i];
        for (0..pages.len) |i| req.page_offsets[i] = pages[i];
        req.event_fd.* = event;
        req.is_done_tokenizing.* = false;
        req.pipeline_offset.* = base + offset - @intFromPtr(pipeline_v);
        req.large_tokens_count.* = 0;
        req.small_tokens_count.* = 0;
        req.page_tokens_count.* = 0;
        req.large_chunks_count.* = 0;
        req.small_chunks_count.* = 0;
        req.page_chunks_count.* = 0;
        req.embeddings_count.* = 0;
        req.offset.* = offset;
    }
};

pub const InferenceEngine = struct {
    pub const MAX_CONCURRENT_REQUESTS = 128;
    pub const GPU_MODELS = 5;
    pub const MAX_SEQUENCE_SIZE = 2048;
    pub const EMBEDDING_DIMENSION = 768;
    pub const MAX_BATCH_SIZE = 1;

    const MAX_SKINNY_INPUT_ID_BYTES = MAX_BATCH_SIZE * MAX_SEQUENCE_SIZE * @sizeOf(u16);
    const MAX_FAT_INPUT_ID_BYTES = MAX_BATCH_SIZE * MAX_SEQUENCE_SIZE * @sizeOf(u64);
    const MAX_ATTENTION_MASK_BYTES = MAX_BATCH_SIZE * MAX_SEQUENCE_SIZE * @sizeOf(u64);
    const MAX_TOKEN_EMBEDDING_BYTES = MAX_BATCH_SIZE * MAX_SEQUENCE_SIZE * EMBEDDING_DIMENSION * @sizeOf(f32);
    const TENSOR_MEMORY_BYTES = MAX_ATTENTION_MASK_BYTES + GPU_MODELS * (MAX_SKINNY_INPUT_ID_BYTES + MAX_FAT_INPUT_ID_BYTES + MAX_TOKEN_EMBEDDING_BYTES);

    shared_memory_allocator: SharedMemoryAllocator,
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

    // notification resources
    events: RingQueue(i32, MAX_CONCURRENT_REQUESTS),

    pub fn init(size: usize) !InferenceEngine {
        return .{
            .high_priority_pipeline = InferencePipeline.init(),
            .low_priority_pipeline = InferencePipeline.init(),
            .shared_memory_allocator = SharedMemoryAllocator.init(size),
            .i = 0,
            .available_gpu_devices = RingQueue(u8, GPU_MODELS).init(),
            .events = RingQueue(i32, MAX_CONCURRENT_REQUESTS).init(),
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
            self.gpu_devices[i].skinny_input_ids_tensor = @ptrFromInt(@intFromPtr(gpu_memory) + offset);
            offset += MAX_SKINNY_INPUT_ID_BYTES;
            self.gpu_devices[i].fat_input_ids_tensor = @ptrFromInt(@intFromPtr(gpu_memory) + offset);
            offset += MAX_FAT_INPUT_ID_BYTES;
            self.gpu_devices[i].token_embeddings_tensor = @ptrFromInt(@intFromPtr(gpu_memory) + offset);
            offset += MAX_TOKEN_EMBEDDING_BYTES;
            self.available_gpu_devices.push(@intCast(i)) catch unreachable;
        }

        for (0..MAX_CONCURRENT_REQUESTS) |_| {
            self.events.push(@intCast(std.os.linux.eventfd(0, std.os.linux.EFD.NONBLOCK))) catch unreachable;
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
                const idx = gpu_device_idx orelse unreachable;
                const gpu_device = &self.gpu_devices[idx];
                self.high_priority_pipeline.embed(base, gpu_device, idx) catch {
                    self.low_priority_pipeline.embed(base, gpu_device, idx) catch {
                        break;
                    };
                };
                self.available_gpu_devices.discard() catch unreachable;
            }
            break;
        }

        self.high_priority_pipeline.notify(base, self) catch {
            self.low_priority_pipeline.notify(base, self) catch {};
        };
    }

    pub fn start_shared_memory_region(self: *InferenceEngine) usize {
        return @intFromPtr(self) + @sizeOf(InferenceEngine);
    }

    fn enqueue_chunked_embedding_request(self: *InferenceEngine, pipeline: *InferencePipeline, text: []const u8, pages: []const u64) !usize {
        const event = self.events.peek() orelse return error.no_event_fds_available;
        defer self.events.discard() catch unreachable;

        const size = ChunkedEmbeddingRequestView.bytes(text.len, pages.len);

        const offset = try self.shared_memory_allocator.alloc(size);
        errdefer self.shared_memory_allocator.free(offset);

        ChunkedEmbeddingRequestView.init_data(self.start_shared_memory_region(), offset, pipeline, event, text, pages);
        try pipeline.enqueue(offset);

        return offset;
    }

    pub fn enqueue_high_priority_chunked_embedding_request(self: *InferenceEngine, text: []const u8, pages: []const u64) !usize {
        return self.enqueue_chunked_embedding_request(&self.high_priority_pipeline, text, pages);
    }

    pub fn enqueue_low_priority_chunked_embedding_request(self: *InferenceEngine, text: []const u8, pages: []const u64) !usize {
        return self.enqueue_chunked_embedding_request(&self.low_priority_pipeline, text, pages);
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
pub extern fn upcast_uint16_to_int64(input: [*]const u16, output: [*]u64, size: usize, stream: CudaStreamHandle) void;
pub extern fn average_token_embeddings(embeddings: [*]f32, batch: usize, size: usize, stream: CudaStreamHandle) void;
