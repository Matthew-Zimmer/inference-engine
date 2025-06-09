const std = @import("std");
const lib = @import("lib");

pub export fn enqueue_high_priority_chunked_embedding_request(eng: *lib.InferenceEngine, text: [*:0]const u8, pages: usize, page_offsets: [*]u64) usize {
    return eng.enqueue_high_priority_chunked_embedding_request(std.mem.span(text), page_offsets[0..pages]) catch std.math.maxInt(usize);
}

pub export fn enqueue_low_priority_chunked_embedding_request(eng: *lib.InferenceEngine, text: [*:0]const u8, pages: usize, page_offsets: [*]u64) usize {
    return eng.enqueue_low_priority_chunked_embedding_request(std.mem.span(text), page_offsets[0..pages]) catch std.math.maxInt(usize);
}

pub export fn load_shared_memory_from_fd(fd: i32, size: usize) [*]u8 {
    return lib.map_and_lock_fd(fd, size);
}

pub export fn unload_shared_memory(addr: [*]const u8, size: usize) void {
    return lib.unlock_and_unmap_fd(addr, size);
}

const EventFdsInfo = extern struct {
    len: usize,
    data: [*]i32,
};

pub export fn event_fds(eng: *lib.InferenceEngine) EventFdsInfo {
    // TODO: this is only ever a snapshot of all event fds in the system
    return .{ .len = lib.InferenceEngine.MAX_CONCURRENT_REQUESTS, .data = @ptrCast(&eng.events.buffer) };
}

pub export fn chunked_request_event_fd(eng: *lib.InferenceEngine, offset: usize) i32 {
    const req = lib.ChunkedEmbeddingRequestView.view(eng.start_shared_memory_region() + offset);

    return req.event_fd.*;
}

const ChunkedEmbeddingResult = extern struct {
    large_chunk_embeddings_len: usize,
    large_chunk_embeddings_data: [*]f32,
    small_chunk_embeddings_len: usize,
    small_chunk_embeddings_data: [*]f32,
    page_chunk_embeddings_len: usize,
    page_chunk_embeddings_data: [*]f32,
};

pub export fn chunked_request_result(eng: *lib.InferenceEngine, offset: usize) ChunkedEmbeddingResult {
    const req = lib.ChunkedEmbeddingRequestView.view(eng.start_shared_memory_region() + offset);

    return .{
        .large_chunk_embeddings_len = req.large_chunks_count.*,
        .large_chunk_embeddings_data = req.large_chunk_embeddings,
        .small_chunk_embeddings_len = req.small_chunks_count.*,
        .small_chunk_embeddings_data = req.small_chunk_embeddings,
        .page_chunk_embeddings_len = req.page_chunks_count.*,
        .page_chunk_embeddings_data = req.page_chunk_embeddings,
    };
}

pub export fn chunked_request_deinit(eng: *lib.InferenceEngine, offset: usize) void {
    eng.shared_memory_allocator.free(offset);
}
