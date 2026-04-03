const std = @import("std");
const log = std.log;

const zml = @import("zml");
const stdx = zml.stdx;

pub const utils = @import("utils.zig");

pub const std_options: std.Options = .{
    .log_level = .info,
};

pub const ResNet18Pipeline = @import("pipeline.zig").ResNet18Pipeline;
pub const ExecutionTrace = @import("resnet18.zig").ExecutionTrace;
pub const ResNet18 = @import("resnet18.zig").ResNet18;

const CliArgs = struct {
    model: []const u8 = "",
    config: []const u8 = "",
    image: []const u8 = "",

    pub const help =
        \\ Usage: resnet-18 [options]
        \\ 
        \\ Run ResNet-18 inference.
        \\
        \\ Options:
        \\   --model <model>    Path to model weights (.safetensors)
        \\   --config <config>  Path to config.json
        \\   --image <image>    Path to input image
    ;
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    const cli_args = stdx.flags.parse(init.minimal.args, CliArgs);

    // Load Image
    const crop_size = 224;
    const tensor_data = try utils.preprocessImage(allocator, cli_args.image, crop_size);
    defer allocator.free(tensor_data);

    // ZML Environment
    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator);

    const model_path = try allocator.dupeZ(u8, cli_args.model);
    defer allocator.free(model_path);
    const config_path = try allocator.dupeZ(u8, cli_args.config);
    defer allocator.free(config_path);

    // Initialize Pipeline
    var pipeline = try ResNet18Pipeline.loadFromFile(allocator, io, platform, .{
        .model_path = model_path,
        .config_path = config_path,
    });
    defer pipeline.deinit();

    // Run generation
    const logits = try pipeline.generate(tensor_data);
    // const logits = try pipeline.generateSequential(tensor_data);
    defer allocator.free(logits);
}
