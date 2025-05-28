const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const lib_mod = b.createModule(.{
        .root_source_file = b.path("src/lib.zig"),
        .target = target,
        .optimize = optimize,
    });

    const runtime_mod = b.createModule(.{
        .root_source_file = b.path("src/public.zig"),
        .target = target,
        .optimize = optimize,
        .pic = true,
    });

    const exe_mod = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    const tests = b.addTest(.{
        .root_source_file = b.path("src/lib.zig"),
        .target = target,
        .optimize = optimize,
    });

    // link cuda + tensor RT to
    // lib inference

    lib_mod.addObjectFile(b.path("trie.o"));
    lib_mod.addObjectFile(b.path("trie_root.o"));
    lib_mod.addObjectFile(b.path("gpu.o"));
    lib_mod.addLibraryPath(b.path("deps/TensorRT-10.10.0.31/lib"));
    lib_mod.addLibraryPath(b.path("deps/cuda_cudart/lib64"));
    lib_mod.linkSystemLibrary("nvinfer", .{});
    lib_mod.linkSystemLibrary("cudart", .{});

    exe_mod.addImport("lib", lib_mod);

    runtime_mod.addImport("lib", lib_mod);

    const version: std.SemanticVersion = .{ .major = 0, .minor = 0, .patch = 0 };

    const exe = b.addExecutable(.{ .name = "inference-engine", .root_module = exe_mod, .version = version });
    const runtime = b.addLibrary(.{ .name = "inference-engine-runtime", .root_module = runtime_mod, .linkage = .dynamic, .version = version });

    exe.linkLibC();
    exe.linkLibCpp();

    runtime.linkLibC();

    b.installArtifact(runtime);
    b.installArtifact(exe);

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&b.addRunArtifact(tests).step);

    const run_cmd = b.addRunArtifact(exe);

    run_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);
}
