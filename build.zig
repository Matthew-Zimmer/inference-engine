const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const libinference = b.addStaticLibrary(.{
        .name = "inference",
        .root_source_file = b.path("src/lib.zig"),
        .target = target,
        .optimize = optimize,
        .version = .{ .major = 0, .minor = 0, .patch = 0 },
    });

    const publibinference = b.addSharedLibrary(.{
        .name = "inference",
        .root_source_file = b.path("src/public.zig"),
        .target = target,
        .optimize = optimize,
        .version = .{ .major = 0, .minor = 0, .patch = 0 },
    });

    const exe = b.addExecutable(.{
        .name = "inference-engine",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    publibinference.linkLibrary(libinference);
    libinference.linkSystemLibrary("c");
    exe.linkLibrary(libinference);
    exe.addObjectFile(b.path("trie.o"));
    exe.addObjectFile(b.path("trie_root.o"));

    b.installArtifact(libinference);
    b.installArtifact(publibinference);
    b.installArtifact(exe);
}
