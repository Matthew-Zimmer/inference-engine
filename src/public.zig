const std = @import("std");
const lib = @import("lib");

pub export fn enqueue_high_priority_chunked_embedding_request(eng: *lib.InferenceEngine, text: [*:0]const u8, pages: usize, page_offsets: [*]u64) usize {
    return eng.enqueue_high_priority_chunked_embedding_request(std.mem.span(text), page_offsets[0..pages]) catch 0;
}

pub export fn enqueue_low_priority_chunked_embedding_request(eng: *lib.InferenceEngine, text: [*:0]const u8, pages: usize, page_offsets: [*]u64) usize {
    return eng.enqueue_low_priority_chunked_embedding_request(std.mem.span(text), page_offsets[0..pages]) catch 0;
}

pub export fn load_shared_memory_from_fd(fd: i32, size: usize) [*]u8 {
    return lib.map_and_lock_fd(fd, size);
}

pub export fn unload_shared_memory(addr: [*]const u8, size: usize) void {
    return lib.unlock_and_unmap_fd(addr, size);
}
