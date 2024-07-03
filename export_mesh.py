import json
import os
from argparse import Namespace

import cv2
import hydra
import mcubes
import numpy as np
import plyfile
import skimage.measure
import torch
import trimesh
import xatlas
from scipy.ndimage import binary_dilation, binary_erosion
from sklearn.neighbors import NearestNeighbors
import nvdiffrast.torch as dr  # 提供了快速的光线追踪和光栅化功能
# from meshutils import *
import torch.nn.functional as F
from packaging import version as pver
from torch import nn

import raymarching

def compute_alpha(net, p, z_s, length=1):

    # net 输入： p:[batch, points, 3] , z:[batch, 1, 256]
    sigma = net.get_sigma(p, z_s)

    alpha = 1 - torch.exp(-sigma * length).view(p.shape[:-1])

    return alpha


@torch.no_grad()
def getDenseAlpha(net, grid_Size, aabb, z_s, step_Size, device="cpu"):
    """
    gridSize是一个3维元组，代表了体素网格在每个维度上的大小。
    aabb是一个2x3的张量，代表了整个模型的坐标范围。
    device是计算设备，例如cpu或gpu。
    stepSize是体素光线穿过体素时的步长。
    compute_alpha是计算某个点alpha值的函数。

    首先，代码使用torch.meshgrid函数生成了一个大小为gridSize的3维网格点，
    每个点对应了密集体素网格中的一个体素。然后，根据网格点计算出了对应的三维坐标点dense_xyz。
    接下来，对于每个格网中的点，循环调用compute_alpha函数，计算其对应的alpha值，最终将alpha值填充到一个3维张量中返回。
    同时，也将对应的坐标点dense_xyz也返回了。
    """

    # 生成采样点 [242, 430, 257, 3]
    samples = torch.stack(torch.meshgrid(
        torch.linspace(0, 1, grid_Size[0]),
        torch.linspace(0, 1, grid_Size[1]),
        torch.linspace(0, 1, grid_Size[2]),
    ), -1).to(device)
    dense_xyz = aabb[0] * (1 - samples) + aabb[1] * samples

    alpha = torch.zeros_like(dense_xyz[..., 0])
    for i in range(grid_Size[0]):
        # compute_alpha 输入【points，3】的xyz点，输出【points】的alpha，再进行变形

        p = dense_xyz[i].view(-1, 3)  # [110510, 3]
        p = p.unsqueeze(0)  # [1, 110510, 3]

        alpha[i] = compute_alpha(net, p, z_s, length=step_Size).view((grid_Size[1], grid_Size[2]))
        # print(alpha[i].shape)  ([430, 257])

    # print(alpha.shape)  # ([242, 430, 257])
    return alpha


def convert_sdf_samples_to_ply(
        pytorch_3d_sdf_tensor,
        ply_filename_out,
        bbox,
        level=0.5,
        offset=None,
        scale=None,
):
    """
    Convert sdf samples to .ply
    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to
    This function adapted from: https://github.com/RobotLocomotion/spartan
    """

    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()
    voxel_size = list((bbox[1] - bbox[0]) / np.array(pytorch_3d_sdf_tensor.shape))

    verts, faces, normals, values = skimage.measure.marching_cubes(
        numpy_3d_sdf_tensor, level=level, spacing=voxel_size
    )
    faces = faces[..., ::-1]  # inverse face orientation

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    # print(mesh_points.shape)
    # print(bbox)
    mesh_points[:, 0] = bbox[0, 0] + verts[:, 0]
    mesh_points[:, 1] = bbox[0, 1] + verts[:, 1]
    mesh_points[:, 2] = bbox[0, 2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    # try writing to the ply file

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    print("saving mesh to %s" % (ply_filename_out))
    ply_data.write(ply_filename_out)

def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')


@torch.no_grad()
def export_stage0(density_grid, density_net, resolution=512, decimate_target=1e5, S=128):
    opt = Namespace(path='data/dance/', O=True, workspace='out/dance/', seed=0, stage=0, ckpt='latest', fp16=True,
                    sdf=True, tcnn=True, test=False, test_no_video=False, test_no_mesh=False, camera_traj='',
                    data_format='colmap', train_split='train', preload=True, random_image_batch=True, downscale=1,
                    bound=1.0, scale=-1, offset=[0, 0, 0], mesh='', enable_cam_near_far=False, enable_cam_center=False,
                    min_near=0.05, enable_sparse_depth=False, enable_dense_depth=False, iters=10000, lr=0.01,
                    lr_vert=0.0001, pos_gradient_boost=1, cuda_ray=True, max_steps=1024, update_extra_interval=16,
                    max_ray_batch=4096, grid_size=128, mark_untrained=True, dt_gamma=0.0, density_thresh=0.001,
                    diffuse_step=1000, diffuse_only=False, background='random', enable_offset_nerf_grad=True, n_eval=5,
                    n_ckpt=50, num_rays=4096, adaptive_num_rays=True, num_points=262144, lambda_density=0,
                    lambda_entropy=0, lambda_tv=0, lambda_depth=0.1, lambda_specular=1e-05, lambda_eikonal=0.1,
                    lambda_rgb=1, lambda_mask=0.1, wo_smooth=False, lambda_lpips=0, lambda_offsets=0.1,
                    lambda_lap=0.001, lambda_normal=0, lambda_edgelen=0, contract=False, patch_size=1,
                    trainable_density_grid=False, color_space='srgb', ind_dim=0, ind_num=500, mcubes_reso=512,
                    env_reso=256, decimate_target=100000.0, mesh_visibility_culling=True, visibility_mask_dilation=50,
                    clean_min_f=16, clean_min_d=10, ssaa=2, texture_size=4096, refine=True,
                    refine_steps_ratio=[0.1, 0.2, 0.3, 0.4, 0.5, 0.7], refine_size=0, refine_decimate_ratio=0,
                    refine_remesh_size=0.02, vis_pose=False, gui=False, W=1000, H=1000, radius=5, fovy=50, max_spp=1,
                    refine_steps=[1000, 2000, 3000, 4000, 5000, 7000])

    grid_size = 256

    # only for the inner mesh inside [-1, 1]
    if resolution is None:
        resolution = grid_size

    device = 'cuda'
    mean_density, density_thresh = 1.0081124305725098, 0.001
    density_thresh = min(mean_density, density_thresh)  # 密度阈值

    # sigmas = np.zeros([resolution] * 3, dtype=np.float32)
    sigmas = torch.zeros([resolution] * 3, dtype=torch.float32, device=device)

    if resolution == grid_size:  # 若分辨率等于grid_size
        # re-map from morton code to regular coords...
        all_indices = torch.arange(resolution ** 3, device=device, dtype=torch.int)
        all_coords = raymarching.morton3D_invert(all_indices).long()
        density_grid = density_grid.reshape(1, -1).to(torch.float)
        # print(density_grid)
        # print(all_coords)
        # print(sigmas)
        sigmas[tuple(all_coords.T)] = density_grid[0]
    else:
        # query，定义了一个S*S*S大小的立方体，XYZ的范围可控制mesh的范围
        X = torch.linspace(-1, 1, resolution).split(S)
        Y = torch.linspace(-1.3, 0.7, resolution).split(S)
        Z = torch.linspace(-1, 1, resolution).split(S)
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = custom_meshgrid(xs, ys, zs)
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)  # [S, 3]

                    with torch.cuda.amp.autocast(enabled=opt.fp16):
                        # val = self.density(pts.to(device))['sigma']  # [S, 1]
                        _, val = density_net(pts.to(device), None)#['sigma']  # [S, 1]
                        # print(val.shape)
                        # exit()
                    sigmas[xi * S: xi * S + len(xs), yi * S: yi * S + len(ys), zi * S: zi * S + len(zs)] = val.reshape(
                        len(xs), len(ys), len(zs))

                    # use the density_grid as a baseline mask (also excluding untrained regions)
        if not opt.sdf:
            mask = torch.zeros([grid_size] * 3, dtype=torch.float32, device=device)
            all_indices = torch.arange(grid_size ** 3, device=device, dtype=torch.int)
            all_coords = raymarching.morton3D_invert(all_indices).long()
            mask[tuple(all_coords.T)] = density_grid[0]
            mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=[resolution] * 3, mode='nearest').squeeze(
                0).squeeze(0)
            mask = (mask > density_thresh)
            sigmas = sigmas * mask

    sigmas = torch.nan_to_num(sigmas, 0)
    sigmas = sigmas.cpu().numpy()

    # import kiui
    # for i in range(254,255):
    #     kiui.vis.plot_matrix((sigmas[..., i]).astype(np.float32))

    if opt.sdf:
        vertices, triangles = mcubes.marching_cubes(sigmas, 0)
    else:
        vertices, triangles = mcubes.marching_cubes(sigmas, density_thresh)

    vertices = vertices / (resolution - 1.0) * 2 - 1
    vertices = vertices.astype(np.float32)
    triangles = triangles.astype(np.int32)

    ### visibility test.
    # if dataset is not None:
    #     visibility_mask = self.mark_unseen_triangles(vertices, triangles, dataset.mvps, dataset.H,
    #                                                  dataset.W).cpu().numpy()
    #     vertices, triangles = remove_masked_trigs(vertices, triangles, visibility_mask,
    #                                               dilation=opt.visibility_mask_dilation)

    # ### reduce floaters by post-processing...
    # vertices, triangles = clean_mesh(vertices, triangles, min_f=opt.clean_min_f, min_d=opt.clean_min_d,
    #                                  repair=True, remesh=False)
    
    # ### decimation
    # if decimate_target > 0 and triangles.shape[0] > decimate_target:
    #     vertices, triangles = decimate_mesh(vertices, triangles, decimate_target, remesh=False)

    mesh = trimesh.Trimesh(vertices, triangles, process=False)
    if not os.path.exists('mesh_stage0'):
        os.makedirs('mesh_stage0')
    mesh.export(f'mesh_stage0/mesh_0.ply')  # 保存第0阶段的粗糙mesh

    # # for the outer mesh [1, inf]
    # if self.bound > 1:
    #
    #     reso = grid_size
    #     target_reso = opt.env_reso
    #     decimate_target //= 2  # empirical...
    #
    #     all_indices = torch.arange(reso ** 3, device=device, dtype=torch.int)
    #     all_coords = raymarching.morton3D_invert(all_indices).cpu().numpy()
    #
    #     # for each cas >= 1
    #     for cas in range(1, self.cascade):
    #         bound = min(2 ** cas, self.bound)
    #         half_grid_size = bound / target_reso
    #
    #         # remap from density_grid
    #         occ = torch.zeros([reso] * 3, dtype=torch.float32, device=device)
    #         occ[tuple(all_coords.T)] = density_grid[cas]
    #
    #         # remove the center (before mcubes)
    #         # occ[reso // 4 : reso * 3 // 4, reso // 4 : reso * 3 // 4, reso // 4 : reso * 3 // 4] = 0
    #
    #         # interpolate the occ grid to desired resolution to control mesh size...
    #         occ = F.interpolate(occ.unsqueeze(0).unsqueeze(0), [target_reso] * 3, mode='trilinear').squeeze(0).squeeze(
    #             0)
    #         occ = torch.nan_to_num(occ, 0)
    #         occ = (occ > density_thresh).cpu().numpy()
    #
    #         vertices_out, triangles_out = mcubes.marching_cubes(occ, 0.5)
    #
    #         vertices_out = vertices_out / (target_reso - 1.0) * 2 - 1  # range in [-1, 1]
    #
    #         # remove the center (already covered by previous cascades)
    #         _r = 0.45
    #         vertices_out, triangles_out = remove_selected_verts(vertices_out, triangles_out,
    #                                                             f'(x <= {_r}) && (x >= -{_r}) && (y <= {_r}) && (y >= -{_r}) && (z <= {_r} ) && (z >= -{_r})')
    #         if vertices_out.shape[0] == 0: continue
    #
    #         vertices_out = vertices_out * (bound - half_grid_size)
    #
    #         # remove the out-of-AABB region
    #         xmn, ymn, zmn, xmx, ymx, zmx = self.aabb_train.cpu().numpy().tolist()
    #         xmn += half_grid_size
    #         ymn += half_grid_size
    #         zmn += half_grid_size
    #         xmx -= half_grid_size
    #         ymx -= half_grid_size
    #         zmx -= half_grid_size
    #         vertices_out, triangles_out = remove_selected_verts(vertices_out, triangles_out,
    #                                                             f'(x <= {xmn}) || (x >= {xmx}) || (y <= {ymn}) || (y >= {ymx}) || (z <= {zmn} ) || (z >= {zmx})')
    #
    #         # clean mesh
    #         vertices_out, triangles_out = clean_mesh(vertices_out, triangles_out, min_f=opt.clean_min_f,
    #                                                  min_d=opt.clean_min_d, repair=False, remesh=False)
    #
    #         if vertices_out.shape[0] == 0: continue
    #
    #         # decimate
    #         if decimate_target > 0 and triangles_out.shape[0] > decimate_target:
    #             vertices_out, triangles_out = decimate_mesh(vertices_out, triangles_out, decimate_target,
    #                                                         optimalplacement=False)
    #
    #         vertices_out = vertices_out.astype(np.float32)
    #         triangles_out = triangles_out.astype(np.int32)
    #
    #         print(f'[INFO] exporting outer mesh at cas {cas}, v = {vertices_out.shape}, f = {triangles_out.shape}')
    #
    #         # if dataset is not None:
    #         #     visibility_mask = self.mark_unseen_triangles(vertices_out, triangles_out, dataset.mvps, dataset.H,
    #         #                                                  dataset.W).cpu().numpy()
    #         #     vertices_out, triangles_out = remove_masked_trigs(vertices_out, triangles_out, visibility_mask,
    #         #                                                       dilation=opt.visibility_mask_dilation)
    #
    #         # vertices_out, triangles_out = clean_mesh(vertices_out, triangles_out, min_f=self.opt.clean_min_f, min_d=self.opt.clean_min_d, repair=False, remesh=False)
    #         mesh_out = trimesh.Trimesh(vertices_out, triangles_out,
    #                                    process=False)  # important, process=True leads to seg fault...
    #         mesh_out.export(os.path.join(save_path, f'mesh_{cas}.ply'))


@torch.no_grad()
def export_stage1(path, geo_feat, h0=2048, w0=2048, png_compression_level=3):
    opt = Namespace(O=True, seed=0, stage=1, ckpt='latest', fp16=True, sdf=True, tcnn=True, test=False, test_no_video=False, test_no_mesh=False, train_split='train', downscale=1, bound=1.0, scale=-1, offset=[0, 0, 0], mesh='', enable_cam_near_far=False, enable_cam_center=False, min_near=0.05, enable_sparse_depth=False, enable_dense_depth=False, iters=5000, lr=0.01, lr_vert=0.0001, pos_gradient_boost=1, cuda_ray=True, max_steps=1024, update_extra_interval=16, max_ray_batch=4096, grid_size=128, mark_untrained=True, dt_gamma=0.0, density_thresh=0.001, diffuse_step=1000, diffuse_only=False, background='random', enable_offset_nerf_grad=True, n_eval=5, n_ckpt=50, num_rays=4096, adaptive_num_rays=True, num_points=262144, lambda_density=0, lambda_entropy=0, lambda_tv=0, lambda_depth=0.1, lambda_specular=1e-05, lambda_eikonal=0.1, lambda_rgb=1, lambda_mask=0.1, wo_smooth=False, lambda_lpips=0, lambda_offsets=0.1, lambda_lap=0.001, lambda_normal=0.01, lambda_edgelen=0, contract=False, patch_size=1, trainable_density_grid=False, color_space='srgb', ind_dim=0, ind_num=500, mcubes_reso=512, env_reso=256, decimate_target=300000.0, mesh_visibility_culling=True, visibility_mask_dilation=5, clean_min_f=8, clean_min_d=5, ssaa=2, texture_size=4096, refine=True, refine_steps_ratio=[0.1, 0.2, 0.3, 0.4, 0.5, 0.7], refine_size=0, refine_decimate_ratio=0, refine_remesh_size=0.01, vis_pose=False, gui=False, W=1000, H=1000, radius=5, fovy=50, max_spp=1, refine_steps=[500, 1000, 1500, 2000, 2500, 3500])

    # png_compression_level: 0 is no compression, 9 is max (default will be 3)
    assert opt.stage > 0
    device = 'cuda'
    cascade = 1
    individual_num = opt.ind_num
    individual_dim = opt.ind_dim
    if individual_dim > 0:
        individual_codes = nn.Parameter(torch.randn(individual_num, individual_dim) * 0.1)  # 生成独立编码张量矩阵
    else:
        individual_codes = None

    glctx = dr.RasterizeGLContext(output_db=False)  # will crash if using GUI...

    # sequentially load cascaded meshes，顺序加载级联网格
    vertices = []
    triangles = []
    v_cumsum = [0]
    f_cumsum = [0]
    for cas in range(cascade):
        # stage0阶段的mesh文件
        _updated_mesh_path = os.path.join('mesh_stage0',
                                              f'mesh_{cas}_updated.ply') if opt.mesh == '' else opt.mesh
        if os.path.exists(_updated_mesh_path) and opt.ckpt != 'scratch':
            mesh = trimesh.load(_updated_mesh_path, force='mesh', skip_material=True, process=False)
        else:  # base (not updated)
            mesh = trimesh.load(os.path.join('mesh_stage0', f'mesh_{cas}.ply'), force='mesh',
                                    skip_material=True, process=False)
        print(f'[INFO] loaded cascade {cas} mesh: {mesh.vertices.shape}, {mesh.faces.shape}')

        vertices.append(mesh.vertices)  # mesh顶点
        triangles.append(mesh.faces + v_cumsum[-1])  # mesh三角面片

        v_cumsum.append(v_cumsum[-1] + mesh.vertices.shape[0])
        f_cumsum.append(f_cumsum[-1] + mesh.faces.shape[0])

    vertices = np.concatenate(vertices, axis=0)  # 合并一个或多个mesh文件的顶点
    triangles = np.concatenate(triangles, axis=0)  # 合并一个或多个mesh文件的三角面片
    v_cumsum = np.array(v_cumsum)
    f_cumsum = np.array(f_cumsum)

    # must put to cuda manually, we don't want these things in the model as buffers...
    vertices = torch.from_numpy(vertices).float().cuda()  # [N, 3]，粗糙mesh的顶点集合
    triangles = torch.from_numpy(triangles).int().cuda()  # 粗糙mesh的三角面片集合

    # learnable offsets for mesh vertex，可学习的mesh角点残差
    vertices_offsets = nn.Parameter(torch.zeros_like(vertices))

    # accumulate error for mesh face，mesh三角面片累计误差
    triangles_errors = torch.zeros_like(triangles[:, 0], dtype=torch.float32).cuda()
    triangles_errors_cnt = torch.zeros_like(triangles[:, 0], dtype=torch.float32).cuda()
    triangles_errors_id = None


    def _export_obj(v, f, h0, w0, ssaa=1, cas=0):  # 导出obj文件
        # v, f: torch Tensor

        v_np = v.cpu().numpy()  # [N, 3]
        f_np = f.cpu().numpy()  # [M, 3]

        print(f'[INFO] running xatlas to unwrap UVs for mesh: v={v_np.shape} f={f_np.shape}')  # 运行xatlas以展开网格的UV

        # unwrap uvs
        atlas = xatlas.Atlas()  # 初始化Atlas
        atlas.add_mesh(v_np, f_np)  # 向atlas中添加mesh的顶点和面片数据
        chart_options = xatlas.ChartOptions()
        chart_options.max_iterations = 0  # disable merge_chart for faster unwrap...；禁用merge_chart以更快地展开。。。
        pack_options = xatlas.PackOptions()
        # pack_options.blockAlign = True
        # pack_options.bruteForce = False
        atlas.generate(chart_options=chart_options, pack_options=pack_options)
        vmapping, ft_np, vt_np = atlas[0]  # [N], [M, 3], [N, 2]

        # vmapping, ft_np, vt_np = xatlas.parametrize(v_np, f_np) # [N], [M, 3], [N, 2]

        vt = torch.from_numpy(vt_np.astype(np.float32)).float().to(device)
        ft = torch.from_numpy(ft_np.astype(np.int64)).int().to(device)

        # render uv maps，渲染uv贴图
        uv = vt * 2.0 - 1.0  # uvs to range [-1, 1]
        uv = torch.cat((uv, torch.zeros_like(uv[..., :1]), torch.ones_like(uv[..., :1])), dim=-1)  # [N, 4]

        if ssaa > 1:
            h = int(h0 * ssaa)
            w = int(w0 * ssaa)
        else:
            h, w = h0, w0

        rast, _ = dr.rasterize(glctx, uv.unsqueeze(0), ft, (h, w))  # [1, h, w, 4]
        xyzs, _ = dr.interpolate(v.unsqueeze(0), rast, f)  # [1, h, w, 3]
        mask, _ = dr.interpolate(torch.ones_like(v[:, :1]).unsqueeze(0), rast, f)  # [1, h, w, 1]

        # masked query，掩码查询
        xyzs = xyzs.view(-1, 3)
        mask = (mask > 0).view(-1)

        feats = torch.zeros(h * w, 6, device=device, dtype=torch.float32)

        if mask.any():
            xyzs = xyzs[mask]  # [M, 3]

            # check individual codes
            if individual_dim > 0:
                ind_code = individual_codes[[0]]
            else:
                ind_code = None

            # batched inference to avoid OOM
            all_feats = []
            head = 0
            while head < xyzs.shape[0]:
                tail = min(head + 640000, xyzs.shape[0])
                with torch.cuda.amp.autocast(enabled=opt.fp16):
                    all_feats.append(geo_feat(xyzs[head:tail], ind_code).float())  # 生成特征贴图
                head += 640000

            feats[mask] = torch.cat(all_feats, dim=0)

        feats = feats.view(h, w, -1)  # 6 channels
        mask = mask.view(h, w)

        # quantize [0.0, 1.0] to [0, 255]
        feats = feats.cpu().numpy()
        feats = (feats * 255).astype(np.uint8)

        ### NN search as a queer antialiasing ...
        mask = mask.cpu().numpy()

        inpaint_region = binary_dilation(mask, iterations=32)  # pad width
        inpaint_region[mask] = 0

        search_region = mask.copy()
        not_search_region = binary_erosion(search_region, iterations=3)
        search_region[not_search_region] = 0

        search_coords = np.stack(np.nonzero(search_region), axis=-1)
        inpaint_coords = np.stack(np.nonzero(inpaint_region), axis=-1)

        knn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(search_coords)
        _, indices = knn.kneighbors(inpaint_coords)

        feats[tuple(inpaint_coords.T)] = feats[tuple(search_coords[indices[:, 0]].T)]

        # do ssaa after the NN search, in numpy
        feats0 = cv2.cvtColor(feats[..., :3], cv2.COLOR_RGB2BGR)  # albedo
        feats1 = cv2.cvtColor(feats[..., 3:], cv2.COLOR_RGB2BGR)  # visibility features

        if ssaa > 1:
            feats0 = cv2.resize(feats0, (w0, h0), interpolation=cv2.INTER_LINEAR)
            feats1 = cv2.resize(feats1, (w0, h0), interpolation=cv2.INTER_LINEAR)

        # cv2.imwrite(os.path.join(path, f'feat0_{cas}.png'), feats0, [int(cv2.IMWRITE_PNG_COMPRESSION), png_compression_level])
        # cv2.imwrite(os.path.join(path, f'feat1_{cas}.png'), feats1, [int(cv2.IMWRITE_PNG_COMPRESSION), png_compression_level])
        cv2.imwrite(os.path.join(path, f'feat0_{cas}.jpg'), feats0)  # 保存漫反射贴图
        cv2.imwrite(os.path.join(path, f'feat1_{cas}.jpg'), feats1)  # 保存高光贴图

        # save obj (v, vt, f /)，保存mesh模型
        obj_file = os.path.join(path, f'mesh_{cas}.obj')
        mtl_file = os.path.join(path, f'mesh_{cas}.mtl')

        print(f'[INFO] writing obj mesh to {obj_file}')
        with open(obj_file, "w") as fp:

            fp.write(f'mtllib mesh_{cas}.mtl \n')

            print(f'[INFO] writing vertices {v_np.shape}')
            for v in v_np:
                fp.write(f'v {v[0]} {v[1]} {v[2]} \n')

            print(f'[INFO] writing vertices texture coords {vt_np.shape}')
            for v in vt_np:
                fp.write(f'vt {v[0]} {1 - v[1]} \n')

            print(f'[INFO] writing faces {f_np.shape}')
            fp.write(f'usemtl defaultMat \n')
            for i in range(len(f_np)):
                fp.write(
                    f"f {f_np[i, 0] + 1}/{ft_np[i, 0] + 1} {f_np[i, 1] + 1}/{ft_np[i, 1] + 1} {f_np[i, 2] + 1}/{ft_np[i, 2] + 1} \n")

        with open(mtl_file, "w") as fp:
            fp.write(f'newmtl defaultMat \n')
            fp.write(f'Ka 1 1 1 \n')
            fp.write(f'Kd 1 1 1 \n')
            fp.write(f'Ks 0 0 0 \n')
            fp.write(f'Tr 1 \n')
            fp.write(f'illum 1 \n')
            fp.write(f'Ns 0 \n')
            fp.write(f'map_Kd feat0_{cas}.jpg \n')

    v = (vertices + vertices_offsets).detach()
    f = triangles.detach()

    for cas in range(cascade):
        cur_v = v[v_cumsum[cas]:v_cumsum[cas + 1]]
        cur_f = f[f_cumsum[cas]:f_cumsum[cas + 1]] - v_cumsum[cas]
        _export_obj(cur_v, cur_f, h0, w0, opt.ssaa, cas)

        # half the texture resolution for remote area.
        if h0 > 2048 and w0 > 2048:
            h0 //= 2
            w0 //= 2
