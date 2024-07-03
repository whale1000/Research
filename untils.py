import cv2
import imageio
import numpy as np
from argparse import Namespace
from skimage.metrics import peak_signal_noise_ratio

trans_t = lambda t : np.array([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1],
], dtype=float)

rot_phi = lambda phi : np.array([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1],
], dtype=float)

rot_theta = lambda th : np.array([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1],
], dtype=float)


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]) @ c2w
    return c2w

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

def get_c2w(i, num_views=60, y=0, print_images_txt=False,mesh_flag=True):
    theta = 2 * np.pi * i / num_views # 绕y轴旋转的角度
    if mesh_flag:
        theta*=-1
        y*=-1
    x = np.cos(theta) * 5.0
    # y = 0
    z = np.sin(theta) * 5.0
    t = np.array([x, y, z])  # 相机平移位姿

    # 计算相机指向矢量
    lookat = np.array([0, 0, 0]) - t
    lookat /= np.linalg.norm(lookat)
    up = np.array([0, -1, 0])

    # 计算旋转矩阵
    zaxis = lookat
    xaxis = np.cross(up, zaxis)
    yaxis = np.cross(zaxis, xaxis)
    R = np.column_stack((-xaxis, -yaxis, zaxis))

    # 将旋转矩阵和平移向量组合成变换矩阵
    c2w = np.eye(4)
    c2w[:3, :3] = R
    c2w[:3, 3] = t

    if print_images_txt:
        from scipy.spatial.transform import Rotation
        r = Rotation.from_matrix(R)
        q=r.as_quat()
        qw, qx, qy, qz = q[0], q[1], q[2], q[3]
        tx, ty, tz = t[0], t[1], t[2]
        print(i, str([qw, qx, qy, qz, tx, ty, tz])[1:-1].replace(',', ''), i, '{}.jpg\n'.format(i))

    return c2w

def get_mvp(H, W, intrinsic, far, near, c2w):
    pose = c2w.copy()
    pose[:3, 2] *= -1  # nerf坐标系和mesh坐标系不一样【Z轴决定是否指向mesh】
    # pose[:3, 1] *= -1  # Y轴决定上下
    pose[:3, 0] *= -1  # X轴决定左右（镜像翻转）
    # print(pose[:3, 3])
    # pose[:3, 3] *= -1  # 平移坐标取反
    # pose[1, 3]=-0.3#-0.25  # 控制相机与模型的视野
    aspect = W / H
    y = H / (2.0 * intrinsic[1])  # fl_y
    projection = np.array([[1 / (y * aspect), 0, 0, 0],
                                 [0, -1 / y, 0, 0],
                                 [0, 0, -(far + near) / (far - near),
                                  -(2 * far * near) / (far - near)],
                                 [0, 0, -1, 0]], dtype=np.float32)  # 计算透视投影矩阵，套公式
    return projection @ np.linalg.inv(pose)  # 投影矩阵乘以相机外参的逆矩阵

def create_dodecahedron_cameras(radius=1, center=np.array([0, 0, 0])):

    vertices = np.array([
        -0.57735,  -0.57735,  0.57735,
        0.934172,  0.356822,  0,
        0.934172,  -0.356822,  0,
        -0.934172,  0.356822,  0,
        -0.934172,  -0.356822,  0,
        0,  0.934172,  0.356822,
        0,  0.934172,  -0.356822,
        0.356822,  0,  -0.934172,
        -0.356822,  0,  -0.934172,
        0,  -0.934172,  -0.356822,
        0,  -0.934172,  0.356822,
        0.356822,  0,  0.934172,
        -0.356822,  0,  0.934172,
        0.57735,  0.57735,  -0.57735,
        0.57735,  0.57735,  0.57735,
        -0.57735,  0.57735,  -0.57735,
        -0.57735,  0.57735,  0.57735,
        0.57735,  -0.57735,  -0.57735,
        0.57735,  -0.57735,  0.57735,
        -0.57735,  -0.57735,  -0.57735,
        ]).reshape((-1,3), order="C")

    length = np.linalg.norm(vertices, axis=1).reshape((-1, 1))
    vertices = vertices / length * radius + center

    # construct camera poses by lookat
    def normalize(x):
        return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)

    # forward is simple, notice that it is in fact the inversion of camera direction!
    forward_vector = normalize(vertices - center)
    # pick a temp up_vector, usually [0, 1, 0]
    up_vector = np.array([0, 1, 0], dtype=np.float32)[None].repeat(forward_vector.shape[0], 0)
    # cross(up, forward) --> right
    right_vector = normalize(np.cross(up_vector, forward_vector, axis=-1))
    # rectify up_vector, by cross(forward, right) --> up
    up_vector = normalize(np.cross(forward_vector, right_vector, axis=-1))

    ### construct c2w
    poses = np.eye(4, dtype=np.float32)[None].repeat(forward_vector.shape[0], 0)
    poses[:, :3, :3] = np.stack((right_vector, up_vector, -forward_vector), axis=-1)
    poses[:, :3, 3] = vertices

    return poses

def imgs2gif(imgs, folder, animation):
    imgs = [(img.cpu().numpy() * 255).astype(np.uint8)[0] for img in imgs]
    imgs = [cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA) for img in imgs]
    imageio.mimsave(f"{folder}/../{animation}.gif", imgs, duration=30)


def get_stage1_opt():
    return Namespace(stage=1, ckpt='latest', fp16=True,
                        sdf=True, tcnn=True, test=False, test_no_video=False, test_no_mesh=False, camera_traj='',
                        data_format='colmap', train_split='train', preload=True, random_image_batch=True, downscale=1,
                        bound=1.0, scale=-1, offset=[0, 0, 0], mesh='', enable_cam_near_far=False,
                        enable_cam_center=False, min_near=0.05, enable_sparse_depth=False, enable_dense_depth=False,
                        iters=5000, lr=0.01, lr_vert=0.0001, pos_gradient_boost=1, cuda_ray=True, max_steps=1024,
                        update_extra_interval=16, max_ray_batch=4096, grid_size=128, mark_untrained=True, dt_gamma=0.0,
                        density_thresh=0.001, diffuse_step=1000, diffuse_only=False, background='random',
                        enable_offset_nerf_grad=True, n_eval=5, n_ckpt=50, num_rays=4096, adaptive_num_rays=True,
                        num_points=262144, lambda_density=0, lambda_entropy=0, lambda_tv=0, lambda_depth=0.1,
                        lambda_specular=1e-05, lambda_eikonal=0.1, lambda_rgb=1, lambda_mask=0.1, wo_smooth=False,
                        lambda_lpips=0, lambda_offsets=0.1, lambda_lap=0.001, lambda_normal=0.01, lambda_edgelen=0,
                        contract=False, patch_size=1, trainable_density_grid=False, color_space='srgb', ind_dim=0,
                        ind_num=500, mcubes_reso=512, env_reso=256, decimate_target=300000.0,
                        mesh_visibility_culling=True, visibility_mask_dilation=5, clean_min_f=8, clean_min_d=5, ssaa=2,
                        texture_size=4096, refine=True, refine_steps_ratio=[0.1, 0.2, 0.3, 0.4, 0.5, 0.7],
                        refine_size=0, refine_decimate_ratio=0, refine_remesh_size=0.01, vis_pose=False, gui=False,
                        radius=5, fovy=50, max_spp=1, refine_steps=[500, 1000, 1500, 2000, 2500, 3500])

def get_stage1_mask(index):
    img = cv2.imread(index, cv2.IMREAD_UNCHANGED)
    # 获取图像的第四个通道作为遮罩
    mask = img[:, :, 3]
    # 进行2倍插值
    mask = cv2.resize(mask, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    return mask.reshape(1, mask.shape[0], mask.shape[1], 1)

def imgs_align(img1, img2):
    print(img1.shape, img2.shape)
    # 水平拼接两张图片
    merged_image = cv2.hconcat([img1, img2])
    # 保存合并后的图片
    cv2.imwrite('merged_image.png', merged_image)
    exit()

    # 提取图像1中的非透明区域轮廓
    contours, _ = cv2.findContours(img1[..., 3], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 找到最大轮廓
    max_contour = max(contours, key=cv2.contourArea)

    # 计算最小外接矩形
    rect1 = cv2.minAreaRect(max_contour)
    box1 = cv2.boxPoints(rect1)

    # 提取图像2中的非透明区域轮廓
    contours, _ = cv2.findContours(img2[..., 3], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 找到最大轮廓
    max_contour = max(contours, key=cv2.contourArea)

    # 计算最小外接矩形
    rect2 = cv2.minAreaRect(max_contour)
    box2 = cv2.boxPoints(rect2)

    # 计算平移向量
    translation_vector = np.mean(box2, axis=0) - np.mean(box1, axis=0)

    # 构建平移矩阵
    M = np.array([[1, 0, translation_vector[0]], [0, 1, translation_vector[1]]], dtype=np.float32)

    # 平移图像1
    translated_img1 = cv2.warpAffine(img1, M, (img2.shape[1], img2.shape[0]))


    # # 计算边缘偏移量
    # x_offset = int(translation_vector[0])
    # y_offset = int(translation_vector[1])
    # # print(x_offset, y_offset)
    # print(x_offset, y_offset)
    # # 创建掩膜
    # mask = np.zeros_like(translated_img1[:, :, 0])
    # mask[:, :x_offset+1] = 255
    # # 将掩膜应用到图像上
    # translated_img1[mask > 0] = (0, 0, 0, 0)
    # # 创建掩膜
    # mask = np.zeros_like(translated_img1[:, :, 0])
    # mask[y_offset-1:, :] = 255
    # # 将掩膜应用到图像上
    # translated_img1[mask > 0] = (0, 0, 0, 0)


    # # 水平拼接掩码
    # mask = np.hstack((translated_img1, img2))
    # # 显示掩码图像
    # cv2.imshow('Mask', translated_img1.astype(np.uint8))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # # 计算PSNR
    # psnr = peak_signal_noise_ratio(translated_img1, img2, data_range=img1.max())
    # # 打印结果
    # print("PSNR:", psnr)

    # if psnr>22:
    #     # 水平拼接两张图片
    #     merged_image = cv2.hconcat([translated_img1, img2])
    #     # 保存合并后的图片
    #     cv2.imwrite('merged_image.png', merged_image)

    return translated_img1, img2

