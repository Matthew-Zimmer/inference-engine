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

    const tests = b.addTest(.{
        .root_source_file = b.path("src/lib.zig"),
        .target = target,
        .optimize = optimize,
    });

    publibinference.linkLibrary(libinference);
    libinference.addObjectFile(b.path("trie.o"));
    libinference.addObjectFile(b.path("trie_root.o"));
    exe.linkLibrary(libinference);
    tests.linkLibrary(libinference);

    b.installArtifact(libinference);
    b.installArtifact(publibinference);
    b.installArtifact(exe);

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&b.addRunArtifact(tests).step);
}
