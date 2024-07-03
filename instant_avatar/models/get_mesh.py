import os

import mcubes
import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from packaging import version as pver

import raymarching
from meshutils import remove_masked_trigs, clean_mesh, decimate_mesh, remove_selected_verts


def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')


@torch.no_grad()
def export_stage0(save_path, resolution=None, decimate_target=1e5, dataset=None, S=128):
    clean_min_f=16
    clean_min_d=10
    grid_size = 128
    sdf = True
    fp16 = True
    # only for the inner mesh inside [-1, 1]
    if resolution is None:
        resolution = grid_size

    device = 'cuda'

    # density_thresh = min(self.mean_density, self.density_thresh)  # 密度阈值
    density_thresh = 0

    # sigmas = np.zeros([resolution] * 3, dtype=np.float32)
    sigmas = torch.zeros([resolution] * 3, dtype=torch.float32, device=device)

    if resolution == grid_size:  # 若分辨率等于grid_size
        # re-map from morton code to regular coords...
        all_indices = torch.arange(resolution ** 3, device=device, dtype=torch.int)
        all_coords = raymarching.morton3D_invert(all_indices).long()
        sigmas[tuple(all_coords.T)] = self.density_grid[0]
    else:
        # query，定义了一个S*S*S大小的立方体
        X = torch.linspace(-1, 1, resolution).split(S)
        Y = torch.linspace(-1, 1, resolution).split(S)
        Z = torch.linspace(-1, 1, resolution).split(S)

        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = custom_meshgrid(xs, ys, zs)
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)  # [S, 3]
                    with torch.cuda.amp.autocast(enabled=fp16):
                        val = self.density(pts.to(device))['sigma']  # [S, 1]
                    sigmas[xi * S: xi * S + len(xs), yi * S: yi * S + len(ys),
                    zi * S: zi * S + len(zs)] = val.reshape(len(xs), len(ys), len(zs))

                    # use the density_grid as a baseline mask (also excluding untrained regions)
        if not sdf:
            mask = torch.zeros([grid_size] * 3, dtype=torch.float32, device=device)
            all_indices = torch.arange(grid_size ** 3, device=device, dtype=torch.int)
            all_coords = raymarching.morton3D_invert(all_indices).long()
            mask[tuple(all_coords.T)] = self.density_grid[0]
            mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=[resolution] * 3, mode='nearest').squeeze(
                0).squeeze(0)
            mask = (mask > density_thresh)
            sigmas = sigmas * mask

    sigmas = torch.nan_to_num(sigmas, 0)
    sigmas = sigmas.cpu().numpy()

    # import kiui
    # for i in range(254,255):
    #     kiui.vis.plot_matrix((sigmas[..., i]).astype(np.float32))

    if sdf:
        vertices, triangles = mcubes.marching_cubes(-sigmas, 0)
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
    #                                               dilation=self.opt.visibility_mask_dilation)

    ### reduce floaters by post-processing...
    vertices, triangles = clean_mesh(vertices, triangles, min_f=clean_min_f, min_d=clean_min_d,
                                     repair=True, remesh=False)

    ### decimation
    if decimate_target > 0 and triangles.shape[0] > decimate_target:
        vertices, triangles = decimate_mesh(vertices, triangles, decimate_target, remesh=False)

    mesh = trimesh.Trimesh(vertices, triangles, process=False)
    mesh.export(os.path.join(save_path, f'mesh_0.ply'))

    # # for the outer mesh [1, inf]
    # if self.bound > 1:
    #
    #     reso = grid_size
    #     target_reso = self.opt.env_reso
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
    #         occ[tuple(all_coords.T)] = self.density_grid[cas]
    #
    #         # remove the center (before mcubes)
    #         # occ[reso // 4 : reso * 3 // 4, reso // 4 : reso * 3 // 4, reso // 4 : reso * 3 // 4] = 0
    #
    #         # interpolate the occ grid to desired resolution to control mesh size...
    #         occ = F.interpolate(occ.unsqueeze(0).unsqueeze(0), [target_reso] * 3, mode='trilinear').squeeze(
    #             0).squeeze(0)
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
    #         vertices_out, triangles_out = clean_mesh(vertices_out, triangles_out, min_f=self.opt.clean_min_f,
    #                                                  min_d=self.opt.clean_min_d, repair=False, remesh=False)
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
    #         if dataset is not None:
    #             visibility_mask = self.mark_unseen_triangles(vertices_out, triangles_out, dataset.mvps, dataset.H,
    #                                                          dataset.W).cpu().numpy()
    #             vertices_out, triangles_out = remove_masked_trigs(vertices_out, triangles_out, visibility_mask,
    #                                                               dilation=self.opt.visibility_mask_dilation)
    #
    #         # vertices_out, triangles_out = clean_mesh(vertices_out, triangles_out, min_f=self.opt.clean_min_f, min_d=self.opt.clean_min_d, repair=False, remesh=False)
    #         mesh_out = trimesh.Trimesh(vertices_out, triangles_out,
    #                                    process=False)  # important, process=True leads to seg fault...
    #         mesh_out.export(os.path.join(save_path, f'mesh_{cas}.ply'))