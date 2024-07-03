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
import torch.nn.functional as F
from untils import get_c2w, get_mvp, get_stage1_opt, imgs2gif, imgs_align, make_rays

@hydra.main(config_path="./confs", config_name="SNARF_NGP")
def main(opt):
    mesh_flag = True
    print("traing of stage1!")
    pl.seed_everything(opt.seed)
    torch.set_printoptions(precision=6)
    print(f"Switch to {os.getcwd()}")

    datamodule = hydra.utils.instantiate(opt.dataset, _recursive_=False)
    # print(opt.model)
    model = hydra.utils.instantiate(opt.model, datamodule=datamodule, _recursive_=False)
    model = model.cuda()
    if not mesh_flag:
        model.eval()

    checkpoints = sorted(glob.glob("checkpoints/*.ckpt"))
    print("Resume from", checkpoints[-1])
    checkpoint = torch.load(checkpoints[-1])
    model.load_state_dict(checkpoint["state_dict"])

    optimizer, scheduler = model.configure_optimizers()
    optimizer = optimizer[0]
    scheduler = scheduler[0]
    
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    animation = "rotation"
    folder = f"animation/{animation}"
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
    error = []
    num_views = 60
    c2w_list = []
    # for i in range(num_views):
    #     # 计算相机位置
    #     c2w_list.append(get_c2w(i+1, num_views))
    # for i in range(30):
    #     c2w_list.append(get_c2w(i,num_views=30,y=0))
    for i in range(60):
        c2w_list.append(get_c2w(i,num_views=60,y=0,mesh_flag=mesh_flag))
    
    for _ in range(10):
        for i in trange(num_views):
            c2w=c2w_list[i]
            # with torch.no_grad():
            rays_o, rays_d = make_rays(K, c2w, H, W)  # 获取光线
            # rays_od = (torch.tensor(rays_o, device=device,dtype=torch.float32),torch.tensor(rays_d, device=device,dtype=torch.float32))

            # if i==num_views-1:
            #     c2w_for_mvp=c2w_list[num_views-1]
            # else:
            #     c2w_for_mvp=c2w_list[num_views-i-2]
            c2w_for_mvp=c2w_list[i]
            # if i==0:
            #     print(c2w_for_mvp)
            mvp = get_mvp(H=H, W=W, intrinsic=[2000, 2000, 540, 540], far=2.5, near=-2.5, c2w=c2w_for_mvp)  # 与隐式模型定义的near-far相反
            # rgb, depth, acc = render_rays(net, rays_od, bound=bound, N_samples=N_samples, device=device, use_view=use_view)
            # batch.keys();['rays_o', 'rays_d', 'betas', 'global_orient', 'body_pose', 'transl', 'near', 'far'])
            batch = {'rays_o': rays_o, 'rays_d':rays_d,
                        'betas':betas.reshape(10), 'global_orient':global_orient[0], 'body_pose':body_pose[0], 'transl':transl[0],
                        'near':np.ones_like(rays_d[..., 0]) * (-2.5), 'far':np.ones_like(rays_d[..., 0]) * 2.5,
                        'mvp':mvp}
            batch = {k: torch.FloatTensor(v).unsqueeze(0).cuda() for k, v in batch.items()}
            batch['index'] = "{}/{}.png".format(folder+'_stage0', int(i+1))
            
            if mesh_flag:
                rgb, depth, alpha = model.render_image_fast(batch, (H, W), 1)  # 渲染像素的rgb和alpha值
            else:
                rgb, depth, alpha, _ = model.render_image_fast(batch, (H, W), 0)  # 渲染像素的rgb和alpha值
            # rgb, depth, alpha = model.training_stage1(batch, (H, W), 1)  # 渲染像素的rgb和alpha值

            if mesh_flag:
                gt_rgb = cv2.imread("{}/{}.png".format('animation/rotation', int(i+1)), cv2.IMREAD_UNCHANGED)
            pred_rgb = (torch.cat([rgb, alpha[..., None]], dim=-1).detach().cpu().numpy() * 255).astype(np.uint8)[0]  # 合并像素rgb和alpha值  # 第一阶段预测的像素颜色
                    
            # gt_rgb, pred_rgb = imgs_align(gt_rgb, pred_rgb)

            # gt_rgb = torch.FloatTensor(gt_rgb).cuda()
            # pred_rgb = torch.cat([rgb, alpha[..., None]], dim=-1) * 255

            # # loss = criterion(pred_rgb, gt_rgb).mean() # [H, W]，计算图片渲染损失
            # loss = F.mse_loss(pred_rgb, gt_rgb, reduction="mean")
            
            # optimizer.zero_grad()  # 在下一次求导之前将保留的grad清空
            # loss.backward()  # 反向传播，计算梯度
            # optimizer.step()  # 应用求导到优化器上去

            # print(loss)
            # model.training_stage1(batch, (H, W), 1)
            # exit()
            # if gt_mask is not None and opt.lambda_mask > 0:
            #     pred_mask = outputs['weights_sum']
            #     loss = loss + opt.lambda_mask * self.criterion(pred_mask.view(-1), gt_mask.view(-1))  # 计算mask损失

            # if self.opt.refine:
            #     self.model.update_triangles_errors(loss.detach())  # 更新三角面片误差

            img = torch.cat([rgb, alpha[..., None]], dim=-1)  # 合并像素rgb和alpha值
            imgs.append(img)
            depths.append(depth)
            if mesh_flag:
                error.append(gt_rgb-pred_rgb)
            else:
                cv2.imwrite("{}/{}.png".format(folder, int(i+1)), (img.cpu().numpy() * 255).astype(np.uint8)[0])
                cv2.imwrite("{}/depth_{}.png".format(folder+'_depth', int(i+1)), (depth.cpu().numpy() * 255).astype(np.uint8)[0])
        
        if mesh_flag:
            errors = [cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA) for img in error]
            imageio.mimsave(f"{folder}/../{animation+'_error'}.gif", errors, duration=30)
        else:
            imgs2gif(imgs=imgs, folder=folder, animation=animation)
        imgs2gif(imgs=depths, folder=folder, animation=animation+'_depth')
        break

if __name__ == "__main__":
    main()
