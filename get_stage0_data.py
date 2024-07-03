import glob
import os

import cv2
import hydra
import imageio
import numpy as np
import pytorch_lightning as pl
import torch
from tqdm import tqdm, trange
from tqdm.auto import tqdm

from untils import get_c2w, get_mvp, imgs2gif, make_rays

@hydra.main(config_path="./confs", config_name="SNARF_NGP")
def main(opt):
    pl.seed_everything(opt.seed)
    torch.set_printoptions(precision=6)
    print(f"Switch to {os.getcwd()}")

    datamodule = hydra.utils.instantiate(opt.dataset, _recursive_=False)
    # print(opt.model)
    model = hydra.utils.instantiate(opt.model, datamodule=datamodule, _recursive_=False)
    model = model.cuda()
    model.eval()
    stage = 0

    checkpoints = sorted(glob.glob("checkpoints/*.ckpt"))
    print("Resume from", checkpoints[-1])
    checkpoint = torch.load(checkpoints[-1])
    model.load_state_dict(checkpoint["state_dict"])

    animation = "rotation"
    folder = f"animation/{animation}_stage0"
    os.makedirs(folder, exist_ok=True)
    os.makedirs(folder+'_depth', exist_ok=True)

    H = W = 1080
    K = np.eye(3)
    K[0, 0] = K[1, 1] = 2000
    K[0, 2] = H // 2
    K[1, 2] = W // 2

    global_orient = np.array([[np.pi, 0, 0]])
    body_pose = np.zeros((1, 69))
    body_pose[:, 2] = 0.5
    body_pose[:, 5] = -0.5
    transl = np.array([[0, 0.20, 0]])
    betas = datamodule.trainset.smpl_params["betas"]

    betas = betas.astype(np.float32)
    body_pose = body_pose.astype(np.float32)
    global_orient = global_orient.astype(np.float32)
    transl = transl.astype(np.float32)

    imgs = []
    depths = []
    # for th in tqdm(np.linspace(0., 360., 120, endpoint=False)):
    #     with torch.no_grad():
    #         c2w = pose_spherical(th, 0., 0.)
    num_views = 60
    # poses = create_dodecahedron_cameras(radius=5)
    # print(poses.shape)
    # exit()
    c2w_list = []

    for i in range(30):
        c2w_list.append(get_c2w(i,num_views=30,y=0))
    for i in range(30):
        c2w_list.append(get_c2w(i,num_views=30,y=2.5))

    # for i in range(num_views):
    #     # 计算相机位置
    #     c2w_list.append(get_c2w(i+1, 0, num_views))
    for i in trange(num_views):
        c2w=c2w_list[i]
        # self.poses, self.pts3d = center_poses(poses, pts3d, self.opt.enable_cam_center)
        # c2w = np.linalg.inv(c2w)
        # c2w[:3, 1:3] *= -1  # 将位姿矩阵中的第二列和第三列乘以-1，即将Y轴和Z轴的方向反转
        # c2w[:3, 2] *= -1  # Z轴表示上下
        # c2w = c2w[[1, 0, 2, 3], :]  # 重新排列位姿矩阵的列顺序，将原来的第一列移至第二列，原来的第零列移至第一列，原来的第二列保持不变，原来的第三列保持不变
        # c2w[:, 2] *= -1  # 将位姿矩阵的第三列乘以-1，即将Z轴的方向反转

        # c2w = poses[i]
        # R_ = c2w[:3,:3]*np.array([[1,0,0],[0,-1,0],[0,0,1]])
        # smpl_R = R_ @ cv2.Rodrigues(global_orient)[0]
        # smpl_R = cv2.Rodrigues(smpl_R)[0].astype(np.float32).reshape(3,)

        with torch.no_grad():
            # print(c2w[:3, 3])
            # R_gt = cv2.Rodrigues(global_orient[0])[0]  # smpl的全局旋转矩阵
            # R_gt = (c2w[:3, :3]*np.array([[1,0,0],[0,1,0],[0,0,1]])) @ R_gt  # 将smpl的全局方向进行旋转
            # R_gt = cv2.Rodrigues(R_gt)[0].astype(np.float32)

            rays_o, rays_d = make_rays(K, c2w, H, W)  # 获取光线
            # rays_od = (torch.tensor(rays_o, device=device,dtype=torch.float32),torch.tensor(rays_d, device=device,dtype=torch.float32))
            
            if i==num_views-1:
                c2w_for_mvp=c2w_list[num_views-1]
            else:
                c2w_for_mvp=c2w_list[num_views-i-2]
            mvp = get_mvp(H=H, W=W, intrinsic=[2000, 2000, 540, 540], far=2.5, near=-2.5, c2w=c2w_for_mvp)
            # rgb, depth, acc = render_rays(net, rays_od, bound=bound, N_samples=N_samples, device=device, use_view=use_view)
            # batch.keys();['rays_o', 'rays_d', 'betas', 'global_orient', 'body_pose', 'transl', 'near', 'far'])
            batch = {'rays_o': rays_o, 'rays_d':rays_d,
                     'betas':betas.reshape(10), 'global_orient':global_orient[0], 'body_pose':body_pose[0], 'transl':transl[0],
                     'near':np.ones_like(rays_d[..., 0]) * (-2.5), 'far':np.ones_like(rays_d[..., 0]) * 2.5,
                     'mvp':mvp}
            batch = {k: torch.FloatTensor(v).unsqueeze(0).cuda() for k, v in batch.items()}
            # for k, v in batch.items():
            #     print(v.shape)
            # exit()
            batch['index'] = i+1
            if stage==0:
                rgb, depth, alpha, _ = model.render_image_fast(batch, (H, W), 0)  # 渲染像素的rgb和alpha值
            else:
                rgb, depth, alpha = model.render_image_fast(batch, (H, W), 1)
            img = torch.cat([rgb, alpha[..., None]], dim=-1)  # 合并像素rgb和alpha值
            imgs.append(img)
            depths.append(depth)
            cv2.imwrite("{}/{}.png".format(folder, int(i+1)), (img.cpu().numpy() * 255).astype(np.uint8)[0])
            cv2.imwrite("{}/depth_{}.png".format(folder+'_depth', int(i+1)), (depth.cpu().numpy() * 255).astype(np.uint8)[0])

    # imgs2gif(imgs=imgs, folder=folder, animation=animation)
    # imgs2gif(imgs=depths, folder=folder, animation=animation+'_depth')
    

if __name__ == "__main__":
    main()
