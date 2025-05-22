const std = @import("std");
const lib = @import("lib.zig");

extern const _binary_trie_bin_start: opaque {};
extern const _binary_trie_root_bin_start: opaque {};
const vocab_trie: [*]const u8 = @ptrCast(&_binary_trie_bin_start);
const vocab_root_trie: [*]const usize = @alignCast(@ptrCast(&_binary_trie_root_bin_start));

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer _ = gpa.deinit();

    const SIZE = 4096 * 100;
    const memfd = lib.create_memory_fd(SIZE);
    defer lib.close_memory_fd(memfd);

    const addr = lib.map_and_lock_fd(memfd, SIZE);
    defer lib.unlock_and_unmap_fd(addr, SIZE);

    const buffer: [32]u8 = .{0} ** 32;
    var writer = std.io.FixedBufferStream([32]u8){ .buffer = buffer, .pos = 0 };
    try std.fmt.format(writer.writer(), "{}", .{memfd});

    const engine: *lib.InferenceEngine = @alignCast(@ptrCast(addr));
    engine.* = try lib.InferenceEngine.init(SIZE, vocab_root_trie, vocab_trie);
    try engine.start(allocator);
    defer engine.deinit();

    const high_priority_pid: std.os.linux.pid_t = @intCast(std.os.linux.fork());
    if (high_priority_pid == 0) {
        _ = lib.map_and_lock_fd(memfd, SIZE);
        const args: [6:null]?[*:0]const u8 = .{ "/home/matthew/inference-engine-2/.venv/bin/python", "-m", "high", @ptrCast(&writer.buffer), "32000000", null };
        const env: [1:null]?[*:0]const u8 = .{null};
        _ = std.os.linux.execve(args[0].?, &args, &env);
        return error.os_execve_error;
    } else if (high_priority_pid > 0) {
        // parent
        var status: u32 = undefined;
        _ = std.os.linux.waitpid(high_priority_pid, &status, 0);
    } else {
        return error.os_fork_error;
    }

    const low_priority_pid: std.os.linux.pid_t = @intCast(std.os.linux.fork());
    if (low_priority_pid == 0) {
        const args: [6:null]?[*:0]const u8 = .{ "/home/matthew/inference-engine-2/.venv/bin/python", "-m", "low", @ptrCast(&writer.buffer), "32000000", null };
        const env: [1:null]?[*:0]const u8 = .{null};
        _ = std.os.linux.execve(args[0].?, &args, &env);
        return error.os_execve_error;
    } else if (low_priority_pid > 0) {
        // parent
        var status: u32 = undefined;
        _ = std.os.linux.waitpid(low_priority_pid, &status, 0);
    } else {
        return error.os_fork_error;
    }

    while (true) {
        try engine.tick();
    }
}
