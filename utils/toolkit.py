import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def count_parameters(model, trainable=False):
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def tensor2numpy(x):
    return x.cpu().data.numpy() if x.is_cuda else x.data.numpy()


def target2onehot(targets, n_classes):
    onehot = torch.zeros(targets.shape[0], n_classes).to(targets.device)
    onehot.scatter_(dim=1, index=targets.long().view(-1, 1), value=1.0)
    return onehot


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def normalize(x):
    x = x / torch.norm(x, dim=1, p=2, keepdim=True)
    return x


def compute_task_accuracies(predicts, targets,max_class, task_size=10):
    """
    predicts: torch.Tensor，预测的类别标签，例如 shape = (N,)
    targets: torch.Tensor，真实标签，例如 shape = (N,)
    task_size: 每个 task 包含的类别数量，默认是 10

    返回：
        一个 list，其中每个元素为对应 task 的准确率
    """
    # 计算总的任务数，假设类别从 0 开始连续编号
    num_tasks = (max_class // task_size) + 1

    task_accuracies = []
    for task in range(num_tasks):
        lower_bound = task * task_size
        upper_bound = (task + 1) * task_size
        # 选取属于当前 task 的样本：真实标签在 [lower_bound, upper_bound)
        mask = (targets >= lower_bound) & (targets < upper_bound)
        total = mask.sum().item()
        if total == 0:
            # 如果当前 task 没有样本，可以选择跳过或者设为 0
            acc = 0.0
        else:
            correct = (predicts[mask] == targets[mask]).sum().item()
            acc = correct / total
        task_accuracies.append(acc)

    return task_accuracies


def accuracy(y_pred, y_true, nb_old, increment=10):
    assert len(y_pred) == len(y_true), "Data length error."
    all_acc = {}
    all_acc["total"] = np.around((y_pred == y_true).sum() * 100 / len(y_true), decimals=2)

    # Grouped accuracy
    for class_id in range(0, np.max(y_true), increment):
        idxes = np.where(np.logical_and(y_true >= class_id, y_true < class_id + increment))[0]
        label = "{}-{}".format(str(class_id).rjust(2, "0"), str(class_id + increment - 1).rjust(2, "0"))
        all_acc[label] = np.around((y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2)

    # Old accuracy
    idxes = np.where(y_true < nb_old)[0]
    all_acc["old"] = (
        0 if len(idxes) == 0 else np.around((y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2))

    # New accuracy
    idxes = np.where(y_true >= nb_old)[0]
    all_acc["new"] = (
        0 if len(idxes) == 0 else np.around((y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2))

    # Harmonic mean of old and new accuracy
    all_acc["harmonic"] = np.around(2 * all_acc["old"] * all_acc["new"] / (all_acc["old"] + all_acc["new"]), decimals=2)
    return all_acc


def split_images_labels(imgs):
    # split trainset.imgs in ImageFolder
    images = []
    labels = []
    for item in imgs:
        images.append(item[0])
        labels.append(item[1])

    return np.array(images), np.array(labels)


def get_attribute(dic, name, default):
    if name in dic:
        return dic[name]
    else:
        print(name, 'not in args, set to', default, ' as default')
        return default


class ClipLoss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def get_logits(self, image_features, text_features, logit_scale):
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features, text_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T

        return logits_per_image, logits_per_text

    def forward(self, image_features, text_features, logit_scale, output_dict=False, re_labels=None):
        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale)

        labels = self.get_ground_truth(device, logits_per_image.shape[0]) if re_labels is None else re_labels

        total_loss = (
                             F.cross_entropy(logits_per_image, labels) +
                             F.cross_entropy(logits_per_text, labels)
                     ) / 2

        return {"contrastive_loss": total_loss} if output_dict else total_loss


def save_tensors(save_name,class_protos):

    # 保存完整张量（包含设备信息）
    torch.save(class_protos, f'{save_name}.pth')  #注意cuda和cpu


# ===================== 数据加载 =====================
def load_tensors(save_name):
    data = torch.load(f'{save_name}.pth', map_location='cpu')
    return data


# ===================== 向量选择 =====================
def select_vectors(targets, n=10):
    """选择笛卡尔坐标系中分布最广的向量"""
    # 使用主成分分析
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    coords = pca.fit_transform(targets)

    # 计算极角
    angles = np.arctan2(coords[:, 1], coords[:, 0])  # 使用PCA后的坐标

    # 选择等角度间隔的样本
    selected_indices = []
    target_angles = np.linspace(-np.pi, np.pi, n, endpoint=False)

    for angle in target_angles:
        diff = np.abs(angles - angle)
        idx = np.argmin(diff)
        if idx not in selected_indices:
            selected_indices.append(idx)

    return selected_indices[:n]

def stratified_sample(features, labels, samples_per_class=50):
    sampled_features = []
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        np.random.shuffle(idx)
        sampled_features.append(features[idx[:samples_per_class]])
    return np.vstack(sampled_features)