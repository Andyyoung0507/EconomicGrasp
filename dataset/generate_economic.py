# 导入库和设置参数
import os
import sys
import torch
import numpy as np
import scipy.io as scio
from PIL import Image
import pdb
import argparse


parser = argparse.ArgumentParser()
# parser.add_argument('--dataset_root', default='/home/xiaoming/dataset/graspnet', help='the root of the GraspNet dataset')
parser.add_argument('--dataset_root', default='/home/axe/Downloads/datasets/GraspNet', help='Dataset root')
parser.add_argument('--camera_type', default='kinect', help='Camera split [realsense/kinect]')
# parser.add_argument('--camera_type', default='realsense', help='Camera split [realsense/kinect]')

cfgs = parser.parse_args()

#  定义数据路径
obj_data_folders = os.path.join(cfgs.dataset_root, "grasp_label")
scenes_data_folders = os.path.join(cfgs.dataset_root, "scenes")
collision_data_folders = os.path.join(cfgs.dataset_root, "collision_label")

# 主程序逻辑
if __name__ == "__main__":
    keeping_views_numbers = 300

    save_data_folders = os.path.join(cfgs.dataset_root, f"economic_grasp_label_{keeping_views_numbers}views")
    if not os.path.exists(save_data_folders):
        os.makedirs(save_data_folders)

    # 遍历场景数据
    # collect the labels from object-level to scene-level
    number = 0
    for label_path in os.listdir(scenes_data_folders):
        print(f"---------The {number} scenes----------")
        meta = scio.loadmat(os.path.join(scenes_data_folders, label_path, cfgs.camera_type, 'meta', '0000.mat'))
        obj_idxs = meta['cls_indexes'].flatten().astype(np.int32)
        object_list = open(os.path.join(scenes_data_folders, label_path, 'object_id_list.txt'), "r")
        scene_collision = np.load(os.path.join(collision_data_folders, label_path, 'collision_labels.npz'))
        # 收集对象级别的抓取标签
        scene_points = []
        scene_pointid = []
        scene_scores = []
        scene_width = []
        for i, obj_idx in enumerate(obj_idxs):
            object_labels = np.load(os.path.join(obj_data_folders, f"{str(obj_idx - 1).zfill(3)}_labels.npz"))
            points = torch.from_numpy(object_labels['points'])
            pointid = torch.ones(points.shape[0]) * i
            width = torch.from_numpy(object_labels['offsets'][:, :, :, :, 2])
            scores = torch.from_numpy(object_labels['scores'])
            # 加载碰撞标签，将发生碰撞的抓取姿态的分数设为0，按照上述文件的处理后得到的数据集中的文件不包含碰撞的抓取Pose了
            # 这也是为什么我和本文作者邮件联系时，他提到了 “模型隐式地学到了碰撞检测和规避的能力”
            collision = torch.from_numpy(scene_collision[f'arr_{i}'])
            scores[collision] = 0
            scores[scores < 0] = 0
            scene_points.append(points)
            scene_pointid.append(pointid)
            scene_scores.append(scores)
            scene_width.append(width)
        scene_points = torch.cat(scene_points, dim=0)
        scene_pointid = torch.cat(scene_pointid, dim=0)
        scene_scores = torch.cat(scene_scores, dim=0)
        scene_width = torch.cat(scene_width, dim=0)

        # filtering labels in bad points
        # 过滤低质量的抓取标签，利用每个点graspness值保留至少包含一个高质量抓取Pose的采样点
        threshold = 0.4
        Ns, V, A, D = scene_scores.size() # Ns表示采样点个数
        grasp_num = V * A * D
        grasp_mask = (scene_scores <= threshold) & (scene_scores > 0) # 过滤掉低质量的抓取标签，这里的分数低表示需要更小的摩擦系数即可完成抓取，质量更好
        grasp_mask = grasp_mask.float()
        grasp_mask = grasp_mask.view(Ns, -1)
        graspness = torch.sum(grasp_mask, dim=-1).float() / grasp_num  # [objects points, 1]
        filter_mask = (graspness > 0)
        ori_number = scene_points.shape[0]
        scene_points = scene_points[filter_mask]
        scene_pointid = scene_pointid[filter_mask]
        scene_scores = scene_scores[filter_mask]
        scene_width = scene_width[filter_mask] # 抓取Pose的宽度数据
        result_number = scene_points.shape[0]
        print(result_number, ori_number) # 此处过滤的是采样点的个数！

        # compute view graspness，计算的是每个视图的graspness
        view_u_threshold = 0.6
        grasp_view_valid_mask = (scene_scores <= view_u_threshold) & (scene_scores > 0)
        grasp_view_valid = grasp_view_valid_mask.float()
        grasp_view_graspness = torch.sum(torch.sum(grasp_view_valid, dim=-1), dim=-1) / 48  # (Ns, V)
        view_graspness_min, _ = torch.min(grasp_view_graspness, dim=-1)  # (Ns)
        view_graspness_max, _ = torch.max(grasp_view_graspness, dim=-1)
        view_graspness_max = view_graspness_max.unsqueeze(-1).expand(-1, 300)  # (Ns, V)
        view_graspness_min = view_graspness_min.unsqueeze(-1).expand(-1,
                                                                     300)  # same shape as batch_grasp_view_graspness
        grasp_view_graspness = (grasp_view_graspness - view_graspness_min) / (
                view_graspness_max - view_graspness_min + 1e-5)  # (Ns, V)

        # nomalize the score
        # 归一化分数并保留视图多样性，确定每个点每个视图下分数最高的抓取的深度和角度，只剩下view维度不固定

        label_mask = (scene_scores > 0) & (scene_width <= 0.1)
        scene_scores[~label_mask] = 0
        po_mask = scene_scores > 0
        scene_scores[po_mask] = 1.1 - scene_scores[po_mask] # 将数据集中分数从摩擦系数转化为分数，分数越高，抓取Pose质量越高

        # only keeping the views
        grasp_score_label = scene_scores
        grasp_width_label = scene_width
        grasp_score_label_max_depth, grasp_score_label_max_depth_idx = grasp_score_label.max(-1) # 首先在深度维度上找分数最大值
        grasp_width_label = grasp_width_label.gather(-1, grasp_score_label_max_depth_idx.unsqueeze(-1)).squeeze(-1)
        grasp_score_label_max_angle, grasp_score_label_max_angle_idx = grasp_score_label_max_depth.max(-1) # 在角度维度上找分数最大值
        scene_depth = grasp_score_label_max_depth_idx.gather(-1, grasp_score_label_max_angle_idx.unsqueeze(-1)).squeeze(
            -1)
        scene_rotations = grasp_score_label_max_angle_idx
        scene_scores = grasp_score_label_max_angle
        scene_width = grasp_width_label.gather(-1, grasp_score_label_max_angle_idx.unsqueeze(-1)).squeeze(-1)

        # further view filtering
        # 使用 torch.topk 保留抓取质量最高的 keeping_views_numbers 个视图。
        values, index = torch.topk(grasp_view_graspness, k=keeping_views_numbers) # 相当于排序，返回的是排序后的值和索引
        scene_rotations = torch.gather(scene_rotations, 1, index) # 根据索引，在scene_rotations中找到对应的值
        scene_depth = torch.gather(scene_depth, 1, index) # 根据索引，在scene_depth中找到对应的值
        scene_scores = torch.gather(scene_scores, 1, index) # 根据索引，在scene_scores中找到对应的值
        scene_width = torch.gather(scene_width, 1, index) # 根据索引，在scene_width中找到对应的值
        scene_top_view_index = index

        # save the results
        # 进一步过滤视图并保存结果
        scene_points = scene_points.numpy()
        grasp_rotations = scene_rotations.numpy().astype(np.uint8)
        grasp_depth = scene_depth.numpy().astype(np.uint8)
        grasp_scores = (scene_scores.numpy() * 10).astype(np.uint8) # 将分数从0-1转换为0-10，为了方便保存为uint8节省空间，否则保存为float32会占用大量空间，直接转变格式会丢失精度
        grasp_widths = (scene_width.numpy() * 1000).astype(np.uint8) # 将米转换为毫米单位, 保存为uint8节省空间
        scene_pointid = scene_pointid.numpy().astype(np.uint8)
        grasp_view_graspness = grasp_view_graspness.numpy()
        grasp_top_view_index = scene_top_view_index.numpy().astype(np.uint16) # 保存为uint16节省空间,view为300，uint16足够,unit8不够,0~255

        np.savez(os.path.join(save_data_folders, f"{label_path}_labels.npz"),
                 points=scene_points, # （Ns, 3）
                 rotations=grasp_rotations, # （Ns, 300）
                 depth=grasp_depth, # （Ns, 300）
                 scores=grasp_scores, # （Ns, 300）
                 widths=grasp_widths, # （Ns, 300）
                 pointid=scene_pointid, # （Ns,)
                 vgraspness=grasp_view_graspness, # （Ns, 300）
                 topview=grasp_top_view_index) # （Ns, 300）

        number = number + 1

    print(f"---------Finishing----------")

