const std = @import("std");

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
};

// TODO: handle utf-8 more gracefully
fn wordpeice_encode(root: [*]const usize, trie: [*]const u8, text: []const u8, tokens: [*]u16) usize {
    var i: usize = 0;
    var root_offset: usize = 0;
    var token_count: usize = 0;
    top: while (i < text.len) {
        const c = text[i];
        switch (c) {
            // skip over whitespace
            // TODO: Think does `root_offset` need to be reset back to 0 here?
            ' ', '\t', '\n', '\r' => i += 1,
            // handle stuff for other characters
            else => {
                const offset = root[c + root_offset];
                // need to check if this logic is needed
                // break out if root char invalid
                if (offset == -1) {
                    tokens[token_count] = UNK;
                    token_count += 1;
                    i += 1;
                    continue;
                }
                var found_tokens: [64]u16 = undefined;
                var found_token_idx: u8 = 1;
                var node = TrieNode.init(&trie[offset]);
                found_tokens[0] = node.id;
                inner: while (i + found_token_idx < text.len) {
                    const b = text[i + found_token_idx];

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
                                    tokens[token_count] = found_tokens[idx];
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
                            tokens[token_count] = UNK;
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
                            found_tokens[found_token_idx] = node.id;
                            found_token_idx += 1;
                            node = TrieNode.init(&trie[node.offsets[j]]);
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
                            tokens[token_count] = found_tokens[idx];
                            token_count += 1;
                            i += idx;
                            // TODO: do we need to set the ## conditionaly here?
                            root_offset = 256;
                            continue :top;
                        }
                    }

                    // will only be here if all found tokens are UNK
                    // only advance "i" by 1 and retry from the start
                    // with a "##" prefix as root
                    tokens[token_count] = UNK;
                    token_count += 1;
                    i += 1;
                    // TODO: do we need to set the ## conditionaly here?
                    root_offset = 256;
                    continue :top;
                }
            },
        }
    }

    return token_count;
}

fn RingQueue(comptime T: type, comptime N: u16) type {
    return struct {
        const Self = @This();

        buffer: [N]T,
        head: u16,
        tail: u16,

        pub fn init() Self {
            return .{ .head = 0, .tail = 0, .buffer = undefined };
        }

        pub fn is_empty(self: *Self) bool {
            return self.head == self.tail;
        }

        pub fn is_full(self: *Self) bool {
            return self.head == (self.tail + 1) % N;
        }

        pub fn peek(self: *Self) ?T {
            if (self.is_empty()) return null;
            return self.buffer[self.head];
        }

        pub fn reserve(self: *Self) !u16 {
            if (self.is_full()) return error.full;
            const idx = self.tail;
            self.tail = (self.tail + 1) % N;
            return idx;
        }

        pub fn insert(self: *Self, idx: u16, val: T) void {
            self.buffer[idx] = val;
        }

        pub fn discard(self: *Self) !void {
            if (self.is_empty()) return error.empty;
            self.head = (self.head + 1) % N;
        }

        pub fn push(self: *Self, val: T) !void {
            if (self.is_full()) return error.full;
            self.buffer[self.tail] = val;
            self.tail = (self.tail + 1) % N;
        }
    };
}

const InferencePipeline = struct {
    const TokenizationQueue = RingQueue(usize, 128);
    const EmbeddingQueue = RingQueue(usize, 1024);

    tokenization_queue: TokenizationQueue,
    embedding_queue: EmbeddingQueue,
    root_trie: [*]const usize,
    trie: [*]const u8,

    pub fn init(root_trie: [*]const usize, trie: [*]const u8) InferencePipeline {
        return .{
            .tokenization_queue = TokenizationQueue.init(),
            .embedding_queue = EmbeddingQueue.init(),
            .root_trie = root_trie,
            .trie = trie,
        };
    }

    pub fn enqueue(self: *InferencePipeline, offset: usize) !void {
        std.debug.print("Pushed to tokenization queue\n", .{});
        return try self.tokenization_queue.push(offset);
    }

    pub fn tokenize(self: *InferencePipeline, base: usize, pool: *std.Thread.Pool) !bool {
        const offset = self.tokenization_queue.peek() orelse return false;
        const slot = self.embedding_queue.reserve() catch return false;
        self.tokenization_queue.discard() catch unreachable;
        std.debug.print("tokenization ready {} {}\n", .{ slot, offset });
        pool.spawn(do_tokenize, .{ self, base, slot, offset }) catch |e| {
            std.debug.print("POOL SPAWN ERROR: {}\n", .{e});
        };
        return true;
    }
};

const EmbeddingRequest = struct {
    size: usize,
    text: []const u8,
    tokens: [*]u16,

    pub fn init(base: usize) EmbeddingRequest {
        const size = @as(*const usize, @ptrFromInt(base)).*;
        const text = @as([*]const u8, @ptrFromInt(base + @sizeOf(usize)))[0..size];
        const tokens = @as([*]u16, @ptrFromInt(base + @sizeOf(usize) + size));

        return .{
            .size = size,
            .text = text,
            .tokens = tokens,
        };
    }
};

fn do_tokenize(pipeline: *InferencePipeline, base: usize, slot: u16, offset: usize) void {
    const req = EmbeddingRequest.init(base + offset);

    const token_count = wordpeice_encode(pipeline.root_trie, pipeline.trie, req.text, req.tokens);
    _ = token_count;

    pipeline.embedding_queue.insert(slot, offset);
}

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
    shared_memory: SharedMemory,
    high_priority_pipeline: InferencePipeline,
    low_priority_pipeline: InferencePipeline,
    pool: std.Thread.Pool = undefined,
    i: usize,

    pub fn init(size: usize, root_trie: [*]const usize, trie: [*]const u8) !InferenceEngine {
        return .{
            .high_priority_pipeline = InferencePipeline.init(root_trie, trie),
            .low_priority_pipeline = InferencePipeline.init(root_trie, trie),
            .shared_memory = SharedMemory.init(size - 0x10000),
            .i = 0,
        };
    }

    pub fn start(self: *InferenceEngine, allocator: std.mem.Allocator) !void {
        try self.pool.init(.{ .allocator = allocator, .n_jobs = 4 });
    }

    pub fn deinit(self: *InferenceEngine) void {
        defer self.pool.deinit();
    }

    pub fn tick(self: *InferenceEngine) !void {
        if (self.i == 100000000) {
            std.debug.print("tick\n", .{});
            self.i = 0;
        }
        self.i += 1;
        const did_tokenize = try self.high_priority_pipeline.tokenize(self.start_shared_memory_region(), &self.pool);
        if (!did_tokenize) _ = try self.low_priority_pipeline.tokenize(self.start_shared_memory_region(), &self.pool);
    }

    fn embedding_request_max_bytes(text_size: usize) usize {
        const n = text_size + @sizeOf(usize);
        return std.mem.alignForward(usize, n, 8);
    }

    fn start_shared_memory_region(self: *InferenceEngine) usize {
        return @intFromPtr(self) + 0x10000;
    }

    fn copy_embedding_request_to_shared_memory(self: *InferenceEngine, text: []const u8, offset: usize) void {
        var off = self.start_shared_memory_region() + offset;
        @as(*usize, @ptrFromInt(off)).* = text.len;
        off += @sizeOf(usize);
        var i: usize = 0;
        while (i < text.len) {
            @as(*u8, @ptrFromInt(off)).* = text[i];
            off += 1;
            i += 1;
        }
    }

    pub fn enqueue_high_priority_embedding_request(self: *InferenceEngine, text: []const u8) !i32 {
        // here we allocate enough memory for the full job (or at least an upper bound)
        const size = InferenceEngine.embedding_request_max_bytes(text.len);
        const offset = try self.shared_memory.alloc(size);
        std.debug.print("Allocated space for a high embed req {} @ {}\n", .{ size, offset });
        // copy the stuff to the offset
        self.copy_embedding_request_to_shared_memory(text, offset);
        std.debug.print("Copied text to allocated shared memory\n", .{});

        try self.high_priority_pipeline.enqueue(offset);
        return 1;
    }

    pub fn enqueue_low_priority_embedding_request(self: *InferenceEngine, text: []const u8) !i32 {
        // here we allocate enough memory for the full job (or at least an upper bound)
        const size = InferenceEngine.embedding_request_max_bytes(text.len);
        const offset = try self.shared_memory.alloc(size);
        std.debug.print("Allocated space for a low embed req {} @ {}\n", .{ size, offset });
        // copy the stuff to the offset
        self.copy_embedding_request_to_shared_memory(text, offset);
        std.debug.print("Copied text to allocated shared memory\n", .{});

        try self.low_priority_pipeline.enqueue(offset);
        return 0;
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
