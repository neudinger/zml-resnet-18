const std = @import("std");
const zml = @import("zml");
const log = std.log;

const ResNet18 = @import("resnet18.zig").ResNet18;

pub const ResNet18Pipeline = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,

    model: ResNet18,
    exe: zml.Exe,
    buffers: zml.Bufferized(ResNet18),

    id2label: std.json.Parsed(std.json.Value),

    pub const LoadOptions = struct {
        model_path: []const u8,
        config_path: []const u8,
    };

    pub fn loadFromFile(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        options: LoadOptions,
    ) !ResNet18Pipeline {
        var registry = try zml.safetensors.TensorRegistry.fromPath(allocator, io, options.model_path);
        defer registry.deinit();

        var store = zml.io.TensorStore.fromRegistry(allocator, &registry);
        defer store.deinit();

        const model_def = ResNet18.init(store.view());

        const input: zml.Tensor = .init(.{ 1, 3, 224, 224 }, .f32);

        const replicated_sharding = try zml.sharding.replicatedSharding(platform);
        log.info("Compiling model...", .{});
        const exe = try platform.compile(allocator, io, model_def, .forward, .{input}, .{ .shardings = &.{replicated_sharding} });
        errdefer exe.deinit();

        log.info("Loading weights...", .{});
        const buffers = try model_def.load(allocator, io, platform, &store);

        const config_file = try std.Io.Dir.openFile(.cwd(), io, options.config_path, .{ .mode = .read_only });
        defer config_file.close(io);

        const config_stat = try config_file.stat(io);
        const config_content = try allocator.alloc(u8, @intCast(config_stat.size));
        defer allocator.free(config_content);
        _ = try config_file.readPositionalAll(io, config_content, 0);

        const parsed_config = try std.json.parseFromSlice(std.json.Value, allocator, config_content, .{});

        return .{
            .allocator = allocator,
            .io = io,
            .platform = platform,
            .model = model_def,
            .exe = exe,
            .buffers = buffers,
            .id2label = parsed_config,
        };
    }

    pub fn deinit(self: *ResNet18Pipeline) void {
        self.exe.deinit();
        self.id2label.deinit();
    }

    pub fn generate(self: *ResNet18Pipeline, tensor_data: []const f32) ![]f32 {
        const replicated_sharding = try zml.sharding.replicatedSharding(self.platform);
        var input_buffer = try zml.Buffer.fromSlice(self.io, self.platform, zml.Slice.init(zml.Shape.init(.{ 1, 3, 224, 224 }, .f32), std.mem.sliceAsBytes(tensor_data)), replicated_sharding);
        defer input_buffer.deinit();

        log.info("Running inference...", .{});

        var args = try self.exe.args(self.allocator);
        defer args.deinit(self.allocator);
        var results = try self.exe.results(self.allocator);
        defer results.deinit(self.allocator);

        args.set(.{ self.buffers, input_buffer });
        self.exe.call(args, &results);

        var result = results.get(zml.Buffer);
        defer result.deinit();

        const logits = try result.toSliceAlloc(self.allocator, self.io);
        defer logits.free(self.allocator);

        const logits_f32 = logits.items(f32);
        var max_val: f32 = -std.math.inf(f32);
        var max_idx: usize = 0;
        for (logits_f32, 0..) |val, i| {
            if (val > max_val) {
                max_val = val;
                max_idx = i;
            }
        }

        const id2label_obj = self.id2label.value.object.get("id2label").?.object;
        var buf: [32]u8 = undefined;
        const key_str = try std.fmt.bufPrint(&buf, "{d}", .{max_idx});
        const label = id2label_obj.get(key_str).?.string;

        log.info("{s}", .{label});
        std.debug.print("{s}\n", .{label});

        const out_logits = try self.allocator.alloc(f32, logits_f32.len);
        @memcpy(out_logits, logits_f32);
        return out_logits;
    }

    pub fn generateSequential(self: *ResNet18Pipeline, tensor_data: []const f32) ![]f32 {
        // We will execute the model layer by layer.
        // For demonstration, we break it into coarse stages here.
        // A true Nova integration would break this down to the Conv2d level.

        const replicated_sharding = try zml.sharding.replicatedSharding(self.platform);
        var input_buffer = try zml.Buffer.fromSlice(self.io, self.platform, zml.Slice.init(zml.Shape.init(.{ 1, 3, 224, 224 }, .f32), std.mem.sliceAsBytes(tensor_data)), replicated_sharding);
        defer input_buffer.deinit();

        log.info("Running inference (Embedder Stage)...", .{});

        // --- Embedder Execution ---
        const embed_exe = try self.platform.compile(self.allocator, self.io, self.model.embedder, .forward, .{zml.Tensor.init(.{ 1, 3, 224, 224 }, .f32)}, .{ .shardings = &.{replicated_sharding} });
        defer embed_exe.deinit();

        var embed_args = try embed_exe.args(self.allocator);
        defer embed_args.deinit(self.allocator);
        var embed_results = try embed_exe.results(self.allocator);
        defer embed_results.deinit(self.allocator);

        embed_args.set(.{ self.buffers.embedder, input_buffer });
        embed_exe.call(embed_args, &embed_results);
        var x_embed = embed_results.get(zml.Buffer);

        log.info("Running inference (Encoder Stage 0)...", .{});
        const s0_exe = try self.platform.compile(self.allocator, self.io, self.model.encoder.stage0, .forward, .{zml.Tensor.init(x_embed.shape(), .f32)}, .{ .shardings = &.{replicated_sharding} });
        defer s0_exe.deinit();
        var s0_args = try s0_exe.args(self.allocator);
        defer s0_args.deinit(self.allocator);
        var s0_results = try s0_exe.results(self.allocator);
        defer s0_results.deinit(self.allocator);

        s0_args.set(.{ self.buffers.encoder.stage0, x_embed });
        s0_exe.call(s0_args, &s0_results);
        var x_s0 = s0_results.get(zml.Buffer);
        x_embed.deinit();

        log.info("Running inference (Encoder Stage 1)...", .{});
        const s1_exe = try self.platform.compile(self.allocator, self.io, self.model.encoder.stage1, .forward, .{zml.Tensor.init(x_s0.shape(), .f32)}, .{ .shardings = &.{replicated_sharding} });
        defer s1_exe.deinit();
        var s1_args = try s1_exe.args(self.allocator);
        defer s1_args.deinit(self.allocator);
        var s1_results = try s1_exe.results(self.allocator);
        defer s1_results.deinit(self.allocator);

        s1_args.set(.{ self.buffers.encoder.stage1, x_s0 });
        s1_exe.call(s1_args, &s1_results);
        var x_s1 = s1_results.get(zml.Buffer);
        x_s0.deinit();

        log.info("Running inference (Encoder Stage 2)...", .{});
        const s2_exe = try self.platform.compile(self.allocator, self.io, self.model.encoder.stage2, .forward, .{zml.Tensor.init(x_s1.shape(), .f32)}, .{ .shardings = &.{replicated_sharding} });
        defer s2_exe.deinit();
        var s2_args = try s2_exe.args(self.allocator);
        defer s2_args.deinit(self.allocator);
        var s2_results = try s2_exe.results(self.allocator);
        defer s2_results.deinit(self.allocator);

        s2_args.set(.{ self.buffers.encoder.stage2, x_s1 });
        s2_exe.call(s2_args, &s2_results);
        var x_s2 = s2_results.get(zml.Buffer);
        x_s1.deinit();

        log.info("Running inference (Encoder Stage 3)...", .{});
        const s3_exe = try self.platform.compile(self.allocator, self.io, self.model.encoder.stage3, .forward, .{zml.Tensor.init(x_s2.shape(), .f32)}, .{ .shardings = &.{replicated_sharding} });
        defer s3_exe.deinit();
        var s3_args = try s3_exe.args(self.allocator);
        defer s3_args.deinit(self.allocator);
        var s3_results = try s3_exe.results(self.allocator);
        defer s3_results.deinit(self.allocator);

        s3_args.set(.{ self.buffers.encoder.stage3, x_s2 });
        s3_exe.call(s3_args, &s3_results);
        var x_s3 = s3_results.get(zml.Buffer);
        x_s2.deinit();

        // --- Final Classifier Stage ---
        log.info("Running inference (Classifier Stage)...", .{});
        const ClassifierWrapper = struct {
            classifier_weight: zml.Tensor,
            classifier_bias: zml.Tensor,

            pub fn forward(ctx: @This(), input: zml.Tensor) zml.Tensor {
                var x = input.mean(3).mean(2).reshape(.{ 1, 512 });
                const xw = x.withTags(.{ .b, .c });
                const xw_dot = xw.dot(ctx.classifier_weight, .c);
                return xw_dot.add(ctx.classifier_bias.broad(xw_dot.shape()));
            }
        };
        const classifier = ClassifierWrapper{
            .classifier_weight = self.model.classifier_weight,
            .classifier_bias = self.model.classifier_bias,
        };

        const clf_exe = try self.platform.compile(self.allocator, self.io, classifier, .forward, .{zml.Tensor.init(x_s3.shape(), .f32)}, .{ .shardings = &.{replicated_sharding} });
        defer clf_exe.deinit();
        var clf_args = try clf_exe.args(self.allocator);
        defer clf_args.deinit(self.allocator);
        var clf_results = try clf_exe.results(self.allocator);
        defer clf_results.deinit(self.allocator);

        const clf_buffers = .{
            .classifier_weight = self.buffers.classifier_weight,
            .classifier_bias = self.buffers.classifier_bias,
        };
        clf_args.set(.{ clf_buffers, x_s3 });
        clf_exe.call(clf_args, &clf_results);

        var result = clf_results.get(zml.Buffer);
        defer result.deinit();
        x_s3.deinit();

        const logits = try result.toSliceAlloc(self.allocator, self.io);
        defer logits.free(self.allocator);

        const logits_f32 = logits.items(f32);
        var max_val: f32 = -std.math.inf(f32);
        var max_idx: usize = 0;
        for (logits_f32, 0..) |val, i| {
            if (val > max_val) {
                max_val = val;
                max_idx = i;
            }
        }

        const id2label_obj = self.id2label.value.object.get("id2label").?.object;
        var buf: [32]u8 = undefined;
        const key_str = try std.fmt.bufPrint(&buf, "{d}", .{max_idx});
        const label = id2label_obj.get(key_str).?.string;

        log.info("{s}", .{label});
        std.debug.print("{s}\n", .{label});

        const out_logits = try self.allocator.alloc(f32, logits_f32.len);
        @memcpy(out_logits, logits_f32);
        return out_logits;
    }
};
