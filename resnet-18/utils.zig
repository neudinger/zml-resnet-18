const std = @import("std");
const zml = @import("zml");
const log = std.log;
const c_interface = @import("c");

pub fn preprocessImage(allocator: std.mem.Allocator, image_path_str: []const u8, crop_size: usize) ![]f32 {
    const image_path = try allocator.dupeZ(u8, image_path_str);
    defer allocator.free(image_path);
    
    var img_width: i32 = 0;
    var img_height: i32 = 0;
    var img_channels: i32 = 0;
    const img_data = c_interface.stbi_load(image_path, &img_width, &img_height, &img_channels, 3);
    if (img_data == null) {
        log.err("Failed to load image: {s}", .{image_path});
        return error.ImageLoadFailed;
    }
    defer c_interface.stbi_image_free(img_data);

    const aspect = @as(f32, @floatFromInt(img_width)) / @as(f32, @floatFromInt(img_height));
    var res_w: i32 = 256;
    var res_h: i32 = 256;
    if (img_width < img_height) {
        res_h = @intFromFloat(256.0 / aspect);
    } else {
        res_w = @intFromFloat(256.0 * aspect);
    }

    const resized_data = allocator.alloc(u8, @as(usize, @intCast(res_w * res_h * 3))) catch unreachable;
    defer allocator.free(resized_data);

    const resize_res = c_interface.stbir_resize_uint8_linear(img_data, img_width, img_height, 0, resized_data.ptr, res_w, res_h, 0, c_interface.STBIR_RGB);
    if (resize_res == null) {
        log.err("Failed to resize image", .{});
        return error.ImageResizeFailed;
    }

    // Center crop
    const start_x = @divTrunc(res_w - @as(i32, @intCast(crop_size)), 2);
    const start_y = @divTrunc(res_h - @as(i32, @intCast(crop_size)), 2);

    var tensor_data = allocator.alloc(f32, 3 * crop_size * crop_size) catch unreachable;

    const mean = [_]f32{ 0.485, 0.456, 0.406 };
    const std_ = [_]f32{ 0.229, 0.224, 0.225 };

    for (0..crop_size) |vy| {
        for (0..crop_size) |vx| {
            const y: usize = @intCast(start_y + @as(i32, @intCast(vy)));
            const x: usize = @intCast(start_x + @as(i32, @intCast(vx)));
            
            const src_idx = (y * @as(usize, @intCast(res_w)) + x) * 3;
            
            const r = @as(f32, @floatFromInt(resized_data[src_idx + 0])) / 255.0;
            const g = @as(f32, @floatFromInt(resized_data[src_idx + 1])) / 255.0;
            const b = @as(f32, @floatFromInt(resized_data[src_idx + 2])) / 255.0;

            const r_norm = (r - mean[0]) / std_[0];
            const g_norm = (g - mean[1]) / std_[1];
            const b_norm = (b - mean[2]) / std_[2];

            const dst_idx_r = 0 * (crop_size * crop_size) + vy * crop_size + vx;
            const dst_idx_g = 1 * (crop_size * crop_size) + vy * crop_size + vx;
            const dst_idx_b = 2 * (crop_size * crop_size) + vy * crop_size + vx;

            tensor_data[dst_idx_r] = r_norm;
            tensor_data[dst_idx_g] = g_norm;
            tensor_data[dst_idx_b] = b_norm;
        }
    }
    return tensor_data;
}
