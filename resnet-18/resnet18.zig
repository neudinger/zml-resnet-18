const std = @import("std");
const zml = @import("zml");

const Conv2d = struct {
    weight: zml.Tensor,
    stride: usize,
    padding: usize,

    pub fn init(store: zml.io.TensorStore.View, stride: usize, padding: usize) Conv2d {
        return .{
            .weight = store.createTensor("weight", .{ .out, .c, .kh, .kw }, .{}),
            .stride = stride,
            .padding = padding,
        };
    }

    pub fn forward(self: Conv2d, input: zml.Tensor) zml.Tensor {
        return input.conv2d(self.weight, .{
            .window_strides = &.{ @intCast(self.stride), @intCast(self.stride) },
            .padding = &.{ @intCast(self.padding), @intCast(self.padding), @intCast(self.padding), @intCast(self.padding) },
        });
    }
};

const BatchNorm2d = struct {
    weight: zml.Tensor,
    bias: zml.Tensor,
    running_mean: zml.Tensor,
    running_var: zml.Tensor,

    pub fn init(store: zml.io.TensorStore.View) BatchNorm2d {
        return .{
            .weight = store.createTensor("weight", .{.c}, .{}),
            .bias = store.createTensor("bias", .{.c}, .{}),
            .running_mean = store.createTensor("running_mean", .{.c}, .{}),
            .running_var = store.createTensor("running_var", .{.c}, .{}),
        };
    }

    pub fn forward(self: BatchNorm2d, input: zml.Tensor) zml.Tensor {
        const eps: f32 = 1e-5;
        const c_dim = self.weight.shape().dim(0);
        const view_shape: zml.Shape = zml.Shape.init(.{ 1, c_dim, 1, 1 }, self.weight.dtype());

        const w = self.weight.reshape(view_shape);
        const b = self.bias.reshape(view_shape);
        const mean = self.running_mean.reshape(view_shape);
        const var_ = self.running_var.reshape(view_shape);

        const normalized = input.sub(mean.broad(input.shape())).div(var_.addConstant(eps).sqrt().broad(input.shape()));
        return normalized.mul(w.broad(input.shape())).add(b.broad(input.shape()));
    }
};

const ResNetConvLayer = struct {
    convolution: Conv2d,
    normalization: BatchNorm2d,
    has_activation: bool,

    pub fn init(store: zml.io.TensorStore.View, stride: usize, padding: usize, has_activation: bool) ResNetConvLayer {
        return .{
            .convolution = Conv2d.init(store.withPrefix("convolution"), stride, padding),
            .normalization = BatchNorm2d.init(store.withPrefix("normalization")),
            .has_activation = has_activation,
        };
    }

    pub fn forward(self: ResNetConvLayer, input: zml.Tensor) zml.Tensor {
        var x = self.convolution.forward(input);
        x = self.normalization.forward(x);
        if (self.has_activation) {
            x = x.relu();
        }
        return x;
    }
};

const ResNetShortCut = struct {
    convolution: Conv2d,
    normalization: BatchNorm2d,

    pub fn init(store: zml.io.TensorStore.View, stride: usize) ResNetShortCut {
        return .{
            .convolution = Conv2d.init(store.withPrefix("convolution"), stride, 0),
            .normalization = BatchNorm2d.init(store.withPrefix("normalization")),
        };
    }

    pub fn forward(self: ResNetShortCut, input: zml.Tensor) zml.Tensor {
        const x = self.convolution.forward(input);
        return self.normalization.forward(x);
    }
};

const ResNetBasicLayer = struct {
    layer0: ResNetConvLayer,
    layer1: ResNetConvLayer,
    shortcut: ?ResNetShortCut,

    pub fn init(store: zml.io.TensorStore.View, stride: usize, use_shortcut: bool) ResNetBasicLayer {
        return .{
            .layer0 = ResNetConvLayer.init(store.withPrefix("layer.0"), stride, 1, true),
            .layer1 = ResNetConvLayer.init(store.withPrefix("layer.1"), 1, 1, false),
            .shortcut = if (use_shortcut) ResNetShortCut.init(store.withPrefix("shortcut"), stride) else null,
        };
    }

    pub fn forward(self: ResNetBasicLayer, input: zml.Tensor) zml.Tensor {
        var x = self.layer0.forward(input);
        x = self.layer1.forward(x);

        var residual = input;
        if (self.shortcut) |sc| {
            residual = sc.forward(input);
        }

        return x.add(residual).relu();
    }
};

const ResNetStage = struct {
    layer0: ResNetBasicLayer,
    layer1: ResNetBasicLayer,

    pub fn init(store: zml.io.TensorStore.View, stride: usize, use_shortcut: bool) ResNetStage {
        return .{
            .layer0 = ResNetBasicLayer.init(store.withPrefix("layers.0"), stride, use_shortcut),
            .layer1 = ResNetBasicLayer.init(store.withPrefix("layers.1"), 1, false),
        };
    }

    pub fn forward(self: ResNetStage, input: zml.Tensor) zml.Tensor {
        const x = self.layer0.forward(input);
        return self.layer1.forward(x);
    }
};

const ResNetEncoder = struct {
    stage0: ResNetStage,
    stage1: ResNetStage,
    stage2: ResNetStage,
    stage3: ResNetStage,

    pub fn init(store: zml.io.TensorStore.View) ResNetEncoder {
        return .{
            .stage0 = ResNetStage.init(store.withPrefix("stages.0"), 1, false),
            .stage1 = ResNetStage.init(store.withPrefix("stages.1"), 2, true),
            .stage2 = ResNetStage.init(store.withPrefix("stages.2"), 2, true),
            .stage3 = ResNetStage.init(store.withPrefix("stages.3"), 2, true),
        };
    }

    pub fn forward(self: ResNetEncoder, input: zml.Tensor) zml.Tensor {
        var x = self.stage0.forward(input);
        x = self.stage1.forward(x);
        x = self.stage2.forward(x);
        return self.stage3.forward(x);
    }
};

const ResNetEmbedder = struct {
    embedder: ResNetConvLayer,

    pub fn init(store: zml.io.TensorStore.View) ResNetEmbedder {
        return .{
            .embedder = ResNetConvLayer.init(store.withPrefix("embedder"), 2, 3, true),
        };
    }

    pub fn forward(self: ResNetEmbedder, input: zml.Tensor) zml.Tensor {
        var x = self.embedder.forward(input);
        return x.maxPool2d(.{
            .window_dimensions = .{ 3, 3 },
            .window_strides = .{ 2, 2 },
            .padding = .{ .{ 1, 1 }, .{ 1, 1 } },
        }).values;
    }
};

pub const ResNet18 = struct {
    embedder: ResNetEmbedder,
    encoder: ResNetEncoder,
    classifier_weight: zml.Tensor,
    classifier_bias: zml.Tensor,

    pub fn init(store: zml.io.TensorStore.View) ResNet18 {
        return .{
            .embedder = ResNetEmbedder.init(store.withPrefix("resnet.embedder")),
            .encoder = ResNetEncoder.init(store.withPrefix("resnet.encoder")),
            .classifier_weight = store.createTensor("classifier.1.weight", .{ .out, .c }, .{}),
            .classifier_bias = store.createTensor("classifier.1.bias", .{.out}, .{}),
        };
    }

    pub fn load(
        self: *const ResNet18,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *const zml.io.TensorStore,
    ) !zml.Bufferized(ResNet18) {
        const replicated_sharding = try zml.sharding.replicatedSharding(platform);
        return zml.io.load(ResNet18, self, allocator, io, platform, store, .{
            .parallelism = 1, // resnet is tiny
            .dma_chunks = 1,
            .dma_chunk_size = 16 * 1024 * 1024,
            .shardings = &.{replicated_sharding},
        });
    }

    pub fn forward(self: ResNet18, input: zml.Tensor) zml.Tensor {
        var x = self.embedder.forward(input);
        x = self.encoder.forward(x);

        x = x.mean(3).mean(2).reshape(.{ 1, 512 });
        const xw = x.withTags(.{ .b, .c });
        const w = self.classifier_weight;
        const b = self.classifier_bias;
        const xw_dot = xw.dot(w, .c);
        return xw_dot.add(b.broad(xw_dot.shape()));
    }
};
