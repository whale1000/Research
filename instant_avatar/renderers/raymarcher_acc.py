import logging
import os
from argparse import Namespace

import torch
from torch.utils.cpp_extension import load

from export_mesh import convert_sdf_samples_to_ply, export_stage0, export_stage1
import nvdiffrast.torch as dr  # 提供了快速的光线追踪和光栅化功能
import trimesh
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from untils import get_stage1_mask, get_stage1_opt

from ..models.structures.density_grid import DensityGrid

logger = logging.getLogger("nerf")
logger.propagate = False
handler = logging.FileHandler("nerf.log")
logger.addHandler(handler)

cuda_dir = os.path.join(os.path.dirname(__file__), "cuda")
raymarch_kernel = load(name='raymarch_kernel',
                       extra_cuda_cflags=[],
                       sources=[f'{cuda_dir}/raymarcher.cpp',
                                f'{cuda_dir}/raymarcher.cu'])  # 光线步进内核


def stratified_sampling(N, step_size):
    device = step_size.device
    z = torch.arange(N, device=device) * step_size[..., None]
    z += torch.rand_like(z) * step_size[..., None]
    return z

def composite(sigma_vals, dists, thresh=0):
    # 0 (transparent) <= alpha <= 1 (opaque)
    tau = torch.relu(sigma_vals) * dists
    alpha = 1.0 - torch.exp(-tau)
    if thresh > 0:
        alpha[alpha < thresh] = 0
    # transimittance = torch.cat([torch.ones_like(alpha[..., 0:1]),
                                # torch.exp(-torch.cumsum(tau, dim=-1))], dim=-1)
    transimittance = torch.cat([torch.ones_like(alpha[..., 0:1]),
                                torch.cumprod(1 - alpha + 1e-10, dim=-1)], dim=-1)
    w = alpha * transimittance[..., :-1]
    return w, transimittance

def ray_aabb(o, d, bbox_min, bbox_max):
    t1 = (bbox_min - o) / d
    t2 = (bbox_max - o) / d

    t_min = torch.minimum(t1, t2)
    t_max = torch.maximum(t1, t2)

    near = t_min.max(dim=-1).values
    far = t_max.min(dim=-1).values
    return near, far

# start====================stage1================================
def scale_img_nhwc(x, size, mag='bilinear', min='bilinear'):
    assert (x.shape[1] >= size[0] and x.shape[2] >= size[1]) or (x.shape[1] < size[0] and x.shape[2] < size[1]), "Trying to magnify image in one dimension and minify in the other"
    y = x.permute(0, 3, 1, 2) # NHWC -> NCHW
    if x.shape[1] > size[0] and x.shape[2] > size[1]: # Minification, previous size was bigger
        y = torch.nn.functional.interpolate(y, size, mode=min)
    else: # Magnification
        if mag == 'bilinear' or mag == 'bicubic':
            y = torch.nn.functional.interpolate(y, size, mode=mag, align_corners=True)
        else:
            y = torch.nn.functional.interpolate(y, size, mode=mag)
    return y.permute(0, 2, 3, 1).contiguous() # NCHW -> NHWC

def scale_img_hwc(x, size, mag='bilinear', min='bilinear'):
    return scale_img_nhwc(x[None, ...], size, mag, min)[0]

def scale_img_nhw(x, size, mag='bilinear', min='bilinear'):
    return scale_img_nhwc(x[..., None], size, mag, min)[..., 0]

def scale_img_hw(x, size, mag='bilinear', min='bilinear'):
    return scale_img_nhwc(x[None, ..., None], size, mag, min)[0, ..., 0]

def safe_normalize(x, eps=1e-20):
    return x / torch.sqrt(torch.clamp(torch.sum(x * x, -1, keepdim=True), min=eps))
# end====================stage1================================

class Raymarcher(torch.nn.Module):
    def __init__(self, MAX_SAMPLES: int, MAX_BATCH_SIZE: int) -> None:
        """
        Args:
            MAX_SAMPLES: number of samples per ray
            MAX_BATCH_SIZE: max samples to evaluate per batch 
        """
        super().__init__()

        self.MAX_SAMPLES = MAX_SAMPLES
        self.MAX_BATCH_SIZE = MAX_BATCH_SIZE

        self.aabb = torch.tensor([[-1.25, -1.55, -1.25],
                                  [ 1.25,  0.95,  1.25]]).float().cuda()
        self.density_grid_train = DensityGrid(64, self.aabb)
        self.density_grid_test = DensityGrid(256)

        # start====================stage1================================
        if True:
            self.glctx = dr.RasterizeGLContext(output_db=False)  # will crash if using GUI...

            # sequentially load cascaded meshes，顺序加载级联网格
            vertices = []
            triangles = []
            v_cumsum = [0]
            f_cumsum = [0]
            mesh = ''
            ckpt = 'latest'
            for cas in range(1):
                # stage0阶段的mesh文件
                _updated_mesh_path = os.path.join('mesh_stage0',
                                                  f'mesh_{cas}_updated.ply') if mesh == '' else mesh
                if os.path.exists(_updated_mesh_path) and ckpt != 'scratch':
                    mesh = trimesh.load(_updated_mesh_path, force='mesh', skip_material=True, process=False)
                else:  # base (not updated)
                    mesh = trimesh.load(os.path.join('mesh_stage0', f'mesh_{cas}.ply'),
                                        force='mesh', skip_material=True, process=False)
                print(f'[INFO] loaded cascade {cas} mesh: {mesh.vertices.shape}, {mesh.faces.shape}')

                vertices.append(mesh.vertices)  # mesh顶点
                triangles.append(mesh.faces + v_cumsum[-1])  # mesh三角面片

                v_cumsum.append(v_cumsum[-1] + mesh.vertices.shape[0])
                f_cumsum.append(f_cumsum[-1] + mesh.faces.shape[0])

            vertices = np.concatenate(vertices, axis=0)  # 合并一个或多个mesh文件的顶点
            triangles = np.concatenate(triangles, axis=0)  # 合并一个或多个mesh文件的三角面片
            self.v_cumsum = np.array(v_cumsum)
            self.f_cumsum = np.array(f_cumsum)

            # must put to cuda manually, we don't want these things in the model as buffers...
            self.vertices = torch.from_numpy(vertices).float().cuda()  # [N, 3]，粗糙mesh的顶点集合
            self.triangles = torch.from_numpy(triangles).int().cuda()  # 粗糙mesh的三角面片集合

            # learnable offsets for mesh vertex，可学习的mesh角点残差
            # self.vertices_offsets = nn.Parameter(torch.zeros_like(self.vertices))
            self.vertices_offsets = torch.zeros_like(self.vertices)

            # accumulate error for mesh face，mesh三角面片累计误差
            self.triangles_errors = torch.zeros_like(self.triangles[:, 0], dtype=torch.float32).cuda()
            self.triangles_errors_cnt = torch.zeros_like(self.triangles[:, 0], dtype=torch.float32).cuda()
            self.triangles_errors_id = None
        else:
            self.glctx = None
        # end====================stage1================================

    def __call__(self, rays, model, net, batch, stage, eval_mode=True, noise=0, bg_color=None):
        if eval_mode:  # 如果是评估模式
            return self.render_test(rays, model, bg_color, batch)
        else:  # 训练模式
            if stage == 0:
                return self.render_train(rays, model, noise, bg_color)  # 返回渲染结果
            else:
                return self.render_stage1(rays, model, net.rgb, noise, bg_color, batch)

    @torch.no_grad()
    def render_test(self, rays, model, bg_color, batch):
        device = rays.o.device  # 获取当前运行设备类型

        rays_o = rays.o.reshape(-1, 3)
        rays_d = rays.d.reshape(-1, 3)
        near = rays.near.reshape(-1).clone()
        far = rays.far.reshape(-1)
        # near, far = ray_aabb(rays_o, rays_d, self.density_grid.min_corner, self.density_grid.max_corner)
        N = rays_o.shape[0]  # 光线数量，每根光线对应一个原点

        # property of pixels
        color = torch.zeros(N, 3, device=device)
        depth = torch.zeros(N, device=device)
        no_hit = torch.ones(N, device=device)
        counter = torch.zeros_like(depth)

        # alive indices
        alive = torch.arange(N, device=device)
        step_size = (far - near) / self.MAX_SAMPLES  # 相邻采样点之间的步长

        density_grid = self.density_grid_test
        offset = density_grid.min_corner
        scale = density_grid.max_corner - density_grid.min_corner

        k = 0
        while k < self.MAX_SAMPLES:
            N_alive = len(alive)
            if N_alive == 0: break

            N_step = max(min(self.MAX_BATCH_SIZE // N_alive, self.MAX_SAMPLES), 1)
            pts, d_new, z_new = raymarch_kernel.raymarch_test(rays_o, rays_d, near, far, alive,
                                                              density_grid.density_field, scale, offset,
                                                              step_size, N_step)  # 通过光线步进内核中的算法获取采样点位置，方向和z_new
            counter[alive] += (d_new > 0).sum(dim=-1)
            mask = d_new > 0

            rgb_vals = torch.zeros_like(pts, dtype=torch.float32)  # 初始化一个指定大小的张量矩阵
            sigma_vals = torch.zeros_like(rgb_vals[..., 0], dtype=torch.float32)  # 初始化一个指定大小的张量矩阵
            if mask.any():
                rgb_vals[mask], sigma_vals[mask] = model(pts[mask], None)  # 计算采样点的颜色和密度

            raymarch_kernel.composite_test(rgb_vals, sigma_vals, d_new, z_new, alive,
                                           color, depth, no_hit, 0.01)
            alive = alive[(no_hit[alive] > 1e-4) & (z_new[:, -1] > 0)]
            k += N_step

        # if batch['index']==60:
        #     # bounding_box = torch.cat((density_grid.min_corner.unsqueeze(0),  density_grid.max_corner.unsqueeze(0)), dim=0)
        #     # convert_sdf_samples_to_ply(density_grid.density_field.detach().cpu(), 'sdf_test.ply', bbox=bounding_box.cpu(), level=0.005)
        #     export_stage0(density_grid.density_field, model)
        # export_stage1('mesh_stage1', geo_feat)
        # exit()
        if bg_color is not None:
            bg_color = bg_color.reshape(-1, 3)
            color = color + no_hit[..., None] * bg_color
        else:
            color = color + no_hit[..., None]
        return {
            "rgb_coarse": color.reshape(rays.o.shape),
            "depth_coarse": depth.reshape(rays.near.shape),
            "alpha_coarse": (1-no_hit).reshape(rays.near.shape),
            "counter_coarse": counter.reshape(rays.near.shape)
        }

    def render_train(self, rays, model, noise, bg_color):
        rays_o = rays.o.reshape(-1, 3)  # 光线原点
        rays_d = rays.d.reshape(-1, 3)  # 光线方向
        near = rays.near.reshape(-1)  # 光线近界
        far = rays.far.reshape(-1)  # 光线远界
        N_step = self.MAX_SAMPLES  # 最大采样点数量

        step_size = (far - near) / N_step  # 相邻采样点之间的步长

        density_grid = self.density_grid_train
        offset = density_grid.min_corner
        scale = density_grid.max_corner - density_grid.min_corner

        z_vals = raymarch_kernel.raymarch_train(rays_o, rays_d, near, far,
                                                density_grid.density_field, scale, offset,
                                                step_size, N_step)  # 计算z_vals，采样点与相机之间的距离
        mask = z_vals > 0

        z_vals = z_vals + torch.rand_like(z_vals) * step_size[:, None]
        pts = z_vals[..., None] * rays_d[:, None] + rays_o[:, None]  # 获取采样点集合

        rgb_vals = torch.zeros_like(pts, dtype=torch.float32)
        sigma_vals = -torch.ones_like(rgb_vals[..., 0], dtype=torch.float32) * 1e3

        if mask.sum() > 0:  # 如果mask之和大于0
            rgb_vals[mask], sigma_vals[mask] = model(pts[mask], None)  # 计算mask区域的采样点的颜色和密度【model为DNeRF.py中构造的匿名函数】
        if noise > 0:  # 如果噪声大于0
            sigma_vals = sigma_vals + noise * torch.randn_like(sigma_vals)

        dists = torch.ones_like(sigma_vals) * step_size[:, None]
        weights, transmittance = composite(sigma_vals.reshape(z_vals.shape), dists, thresh=0)  # 光线上逐个采样点的权重和透射率
        no_hit = transmittance[..., -1]

        color = (weights[..., None] * rgb_vals.reshape(pts.shape)).sum(dim=-2)  # 计算像素颜色
        if bg_color is not None:  # 如果背景颜色非None
            bg_color = bg_color.reshape(-1, 3)
            color = color + no_hit[..., None] * bg_color
        else:
            color = color + no_hit[..., None]

        depth = (weights * z_vals).sum(dim=-1)  # 计算深度
        return {
            "rgb_coarse": color.reshape(rays.o.shape),
            "depth_coarse": depth.reshape(rays.near.shape),
            "alpha_coarse": (weights.sum(-1)).reshape(rays.near.shape),
            "weight_coarse": weights.reshape(*rays.near.shape, -1),
        }

    # phase 2
    def render_stage1(self, rays, model, net_rgb, noise, bg_color, batch):
        mvp = batch['mvp'].reshape(4, 4)
        # rays_o, rays_d, mvp, h0, w0, index=None, bg_color=None, shading='full', **kwargs
        # prefix = rays_d.shape[:-1]
        # rays_d = rays_d.contiguous().view(-1, 3)
        
        opt = get_stage1_opt()

        rays_o = rays.o.reshape(-1, 3)  # 光线原点
        rays_d = rays.d.reshape(-1, 3)  # 光线方向
        h0, w0 = 1080, 1080
        bg_color = None
        shading = 'full'

        N = rays_d.shape[0]  # N = B * N, in fact
        device = rays_d.device

        # do super-sampling
        if opt.ssaa > 1:
            h = int(h0 * opt.ssaa)
            w = int(w0 * opt.ssaa)
            # interpolate rays_d when ssaa > 1 ...
            dirs = rays_d.view(h0, w0, 3)
            dirs = scale_img_hwc(dirs, (h, w), mag='nearest').view(-1, 3).contiguous()
        else:
            h, w = h0, w0
            dirs = rays_d.contiguous()

        dirs = safe_normalize(dirs)

        # mix background color
        if bg_color is None:
            bg_color = 1

        # [N, 3] to [h, w, 3]
        if torch.is_tensor(bg_color) and len(bg_color.shape) == 2:
            bg_color = bg_color.view(h0, w0, 3)

        # if self.individual_dim > 0:
        #     if self.training:
        #         ind_code = self.individual_codes[index]
        #     # use a fixed ind code for the unknown test data.
        #     else:
        #         ind_code = self.individual_codes[[0]]
        # else:
        #     ind_code = None
        ind_code = None

        results = {}

        # mvp = torch.tensor([[-4.8670,  1.7890,  0.3696,  0.1483],
        #               [-0.5172,  0.0208, -6.9120, -0.6932],
        #               [-0.3434, -0.9390,  0.0229,  1.4826],
        #               [-0.3434, -0.9389,  0.0229,  1.5824]], device='cuda:0')

        vertices = self.vertices + self.vertices_offsets  # [N, 3]，mesh网格顶点调整
        # print(vertices.min(), vertices.max())
        vertices_clip = torch.matmul(F.pad(vertices.float(), pad=(0, 1), mode='constant', value=1.0),
                                     torch.transpose(mvp.float(), 0, 1)).float().unsqueeze(0)  # [1, N, 4]，顶点修剪（mvp存在问题）
        # print(vertices_clip.min(), vertices_clip.max())
        # print(vertices_clip.min(), vertices_clip.max(), self.triangles.min(), self.triangles.max(), (h, w))
        rast, _ = dr.rasterize(self.glctx, vertices_clip, self.triangles, (h, w))  # 光栅化处理（元素全为0，存在问题！需进一步确定mesh是否做了标准化或归一化处理）
        # print(rast.min(), rast.max())
        xyzs, _ = dr.interpolate(vertices.unsqueeze(0), rast, self.triangles)  # [1, H, W, 3]
        mask, _ = dr.interpolate(torch.ones_like(vertices[:, :1]).unsqueeze(0), rast, self.triangles)  # [1, H, W, 1]
        mask_flatten = (mask > 0).view(-1).detach()  # 与隐式模型的mask不一致
        # mask1 = torch.Tensor(get_stage1_mask(batch['index'])).cuda()
        # mask1_flatten = (mask1 > 0).view(-1).detach()
        xyzs = xyzs.view(-1, 3)#.cpu().numpy()  # 3D点坐标全为0，存在异常

        # random noise to make appearance more robust，添加噪声使得外观学习更具鲁棒性
        if self.training:
            xyzs = xyzs + torch.randn_like(xyzs) * 1e-3

        rgbs = torch.zeros(h * w, 3, device=device, dtype=torch.float32)

        if mask_flatten.any():
            with torch.cuda.amp.autocast(enabled=opt.fp16):
                # mask_rgbs, masked_specular = net_rgb(
                #     xyzs[mask_flatten] if opt.enable_offset_nerf_grad else xyzs[mask_flatten].detach(),
                #     dirs[mask_flatten], ind_code, shading)  # stage1阶段计算采样点颜色及其高光颜色

                mask_rgbs, masked_specular = net_rgb(
                    xyzs[mask_flatten] if opt.enable_offset_nerf_grad else xyzs[mask_flatten].detach(),
                    dirs[mask_flatten], ind_code, 'diffuse')  # stage1阶段计算采样点颜色及其高光颜色

                # mask_rgbs, mask_sigma = model(xyzs[mask_flatten], None)  # model为DNeRF.py中定义的匿名函数【并未用到dir，且对采样点做了形变】

            rgbs[mask_flatten] = mask_rgbs.float()

        rgbs = rgbs.view(1, h, w, 3)
        alphas = mask.float()

        # 对alphas和rgbs进行抗锯齿处理，pos_gradient_boost用于指定在抗锯齿处理过程中应用的梯度增强量
        alphas = dr.antialias(alphas, rast, vertices_clip, self.triangles,
                              pos_gradient_boost=opt.pos_gradient_boost).squeeze(0).clamp(0, 1)
        rgbs = dr.antialias(rgbs, rast, vertices_clip, self.triangles,
                            pos_gradient_boost=opt.pos_gradient_boost).squeeze(0).clamp(0, 1)

        image = alphas * rgbs
        depth = alphas * rast[0, :, :, [2]]
        T = 1 - alphas

        # trig_id for updating trig errors
        trig_id = rast[0, :, :, -1] - 1  # [h, w]

        # ssaa
        if opt.ssaa > 1:
            image = scale_img_hwc(image, (h0, w0))
            depth = scale_img_hwc(depth, (h0, w0))
            T = scale_img_hwc(T, (h0, w0))
            trig_id = scale_img_hw(trig_id.float(), (h0, w0), mag='nearest', min='nearest')

        self.triangles_errors_id = trig_id

        image = image + T * bg_color

        # image = image.view(*prefix, 3)
        # depth = depth.view(*prefix)

        # results['depth'] = depth
        # results['image'] = image
        # results['weights_sum'] = 1 - T

        return {
            "rgb_coarse": image.reshape(rays.o.shape),
            "depth_coarse": depth.reshape(rays.near.shape),
            "alpha_coarse": (1-T).reshape(rays.near.shape),
            # "weight_coarse": weights.reshape(*rays.near.shape, -1),
        }