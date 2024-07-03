import glob
import os
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import hydra
from tqdm import tqdm
import imageio


def get_ray_directions(H, W):
    x, y = np.meshgrid(np.arange(W), np.arange(H), indexing="xy")
    xy = np.stack([x, y, np.ones_like(x)], axis=-1)
    return xy

def make_rays(K, c2w, H, W):
    xy = get_ray_directions(H, W).reshape(-1, 3).astype(np.float32)
    d_c = xy @ np.linalg.inv(K).T
    d_w = d_c @ c2w[:3, :3].T
    d_w = d_w / np.linalg.norm(d_w, axis=1, keepdims=True)
    o_w = np.tile(c2w[:3, 3], (len(d_w), 1))
    return o_w.astype(np.float32), d_w.astype(np.float32)

class AnimateDataset(torch.utils.data.Dataset):
    def __init__(self, num_frames, betas, downscale=1):
        H = 1080
        W = 1080

        K = np.eye(3)
        K[0, 0] = K[1, 1] = 2000
        K[0, 2] = H // 2
        K[1, 2] = W // 2
        
        if downscale > 1:  # 下采样因子
            H = H // downscale
            W = W // downscale
            K[:2] /= downscale
        self.H = H
        self.W = W

        c2w = np.eye(4)  # 相机到世界坐标系的转换矩阵【np.eye(4)表示相机坐标系和世界坐标系重合，相机初始位置在世界坐标系原点】

        # 设置目标位置
        target_position = np.array([0, 2.5, 5])  # 使得相机始终指向目标位置
        # 计算方向向量
        direction_vector = target_position / np.linalg.norm(target_position)
        # 计算旋转矩阵
        rotation_matrix = np.eye(4)
        rotation_matrix[:3, 2] = direction_vector
        rotation_matrix[:3, 0] = np.cross(np.array([0, 1, 0]), rotation_matrix[:3, 2])
        rotation_matrix[:3, 0] /= np.linalg.norm(rotation_matrix[:3, 0])
        rotation_matrix[:3, 1] = np.cross(rotation_matrix[:3, 2], rotation_matrix[:3, 0])
        # 应用旋转矩阵到 c2w
        c2w[:3, :3] = rotation_matrix[:3, :3]

        self.rays_o, self.rays_d = make_rays(K, c2w, H, W)  # 获取光线

        global_orient = np.array([[np.pi, 0, 0]])
        body_pose = np.zeros((1, 69))
        body_pose[:, 2] = 0.5
        body_pose[:, 5] = -0.5
        transl = np.array([[0, 2.5, 5]])

        self.betas = betas.astype(np.float32)
        self.body_pose = body_pose.astype(np.float32)
        self.global_orient = global_orient.astype(np.float32)
        self.transl = transl.astype(np.float32)
        self.num_frames = num_frames  # 图片数量

    def __len__(self):
        return self.num_frames  # 返回图片数量，即数据集长度

    def __getitem__(self, idx):
        # prepare NeRF data
        rays_o = self.rays_o
        rays_d = self.rays_d

        datum = {
            # NeRF
            "rays_o": rays_o,
            "rays_d": rays_d,

            # SMPL parameters
            "betas": self.betas.reshape(10),
            "global_orient": self.global_orient[0],  # 初始smpl的6dof位姿，始终不变
            "body_pose": self.body_pose[0],
            "transl": self.transl[0],
        }

        angle = 2 * np.pi * idx / self.num_frames  # 角度
        R = cv2.Rodrigues(np.array([0, angle, 0]))[0]
        R_gt = cv2.Rodrigues(datum["global_orient"])[0]  # smpl的全局旋转矩阵
        R_gt = R @ R_gt  # 将smpl的全局方向进行旋转

        R_ = cv2.Rodrigues(np.array([0, -angle, 0]))[0]
        cam_R_gt = R_ @ cv2.Rodrigues(datum["global_orient"])[0]  # 相机旋转【逆时针】
        cam_t_gt = R_ @ datum["transl"]  # 相机平移【逆时针】
        datum['cam_R'] = cam_R_gt
        datum['cam_t'] = cam_t_gt

        R_gt = cv2.Rodrigues(R_gt)[0].astype(np.float32)  # 将旋转矩阵转为向量，smpl模型全局方向不断变化，即相机不动，smpl模型旋转变化【顺时针】
        datum["global_orient"] = R_gt.reshape(3)  # 记录每个相机视角对应的smpl全局方向

        # distance from camera (0, 0, 0) to midhip，从相机（0，0，0）到臀部中部的距离
        datum["near"] = np.ones_like(rays_d[..., 0]) * 0
        datum["far"] = np.ones_like(rays_d[..., 0]) * 10
        return datum


@hydra.main(config_path="./confs", config_name="SNARF_NGP")
def main(opt):
    pl.seed_everything(opt.seed)
    torch.set_printoptions(precision=6)
    print(f"Switch to {os.getcwd()}")

    datamodule = hydra.utils.instantiate(opt.dataset, _recursive_=False)
    model = hydra.utils.instantiate(opt.model, datamodule=datamodule, _recursive_=False)
    model = model.cuda()
    model.eval()

    checkpoints = sorted(glob.glob("checkpoints/*.ckpt"))
    print("Resume from", checkpoints[-1])
    checkpoint = torch.load(checkpoints[-1])
    model.load_state_dict(checkpoint["state_dict"])

    num_frames = 60
    dataset = AnimateDataset(num_frames,
                             betas=datamodule.trainset.smpl_params["betas"],
                             downscale=1)
    datamodule.testset.image_shape = (dataset.H, dataset.W)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)  # 数据加载器

    animation = "rotation"
    folder = f"animation/{animation}/"
    os.makedirs(folder, exist_ok=True)

    with torch.inference_mode():
        imgs = []
        for i, batch in tqdm(enumerate(dataloader)):  # 遍历数据块
            # batch.keys();['rays_o', 'rays_d', 'betas', 'global_orient', 'body_pose', 'transl', 'near', 'far'])
            batch = {k: v.cuda() for k, v in batch.items()}
            batch['index'] = i+1
            rgb, _, alpha, _ = model.render_image_fast(batch, (dataset.H, dataset.W), 0)  # 渲染像素的rgb和alpha值
            img = torch.cat([rgb, alpha[..., None]], dim=-1)  # 合并像素rgb和alpha值
            imgs.append(img)
            cv2.imwrite("{}/{}.png".format(folder, i+1), (img.cpu().numpy() * 255).astype(np.uint8)[0])

    imgs = [(img.cpu().numpy() * 255).astype(np.uint8)[0] for img in imgs]
    imgs = [cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA) for img in imgs]
    imageio.mimsave(f"{folder}/../{animation}.gif", imgs, duration=30)

if __name__ == "__main__":
    main()


#             # IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
#             r = Rotation.from_matrix(batch['cam_R'].cpu().numpy())
#             q = r.as_quat()[0]
#             QW, QX, QY, QZ = q[0], q[1], q[2], q[3]
#             TX, TY, TZ = Rt[:3, 3][0] ,Rt[:3, 3], Rt[:3, 3][2]
#             f_images.append([i+1, QW, QX, QY, QZ, TX, TY, TZ, i+1, '{}.png'.format(i)])