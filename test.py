import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models import resnet18
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")
print(f"using {device}")

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models import resnet18
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# 设置设备


# 加载图片
screenshot = Image.open("screenshot.png").convert("RGB")
template = Image.open("wifi.png").convert("RGB")

# 获取尺寸
W, H = screenshot.size
w, h = template.size

# 预处理 transform（不 resize！）
transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

# 模板 tensor
template_tensor = transform(template).unsqueeze(0).to(device)  # [1, 3, h, w]

# 滑动窗口提取 patch：RGB 空间滑窗
step = 4  # 步长可调，越小越精细但越慢
patches = []
positions = []

for y in range(0, H - h + 1, step):
    for x in range(0, W - w + 1, step):
        patch = screenshot.crop((x, y, x + w, y + h))
        patches.append(transform(patch))
        positions.append((x, y))

# 通过 batch_size 控制每次处理的 patch 数量
batch_size = 64  # 每个批次的 patch 数量
patches = torch.stack(patches).to(device)  # [N, 3, h, w]

# 合并 template 和所有 patches 进入 backbone
backbone = resnet18(pretrained=True).to(device).eval()
backbone = torch.nn.Sequential(*list(backbone.children())[:-1])  # 去掉fc层，输出 [B, 512, 1, 1]

# 使用分批处理
all_patch_feats = []
with torch.no_grad():
    for i in range(0, len(patches), batch_size):
        batch_patches = patches[i:i+batch_size]  # 选择当前批次
        # 处理当前批次
        patch_feats = backbone(batch_patches).squeeze(-1).squeeze(-1)  # [batch_size, 512]
        all_patch_feats.append(patch_feats)

    # 合并所有 batch 特征
    patch_feats = torch.cat(all_patch_feats, dim=0)  # [N, 512]
    
    # template 特征
    template_feat = backbone(template_tensor).squeeze().unsqueeze(0)  # [1, 512]

# 归一化后做余弦相似度
patch_feats = F.normalize(patch_feats, dim=1)
template_feat = F.normalize(template_feat, dim=1)
sims = torch.matmul(patch_feats, template_feat.T).squeeze()  # [N]

# 找最大位置
best_idx = sims.argmax().item()
best_x, best_y = positions[best_idx]
best_score = sims[best_idx].item()

# 可视化
fig, ax = plt.subplots()
ax.imshow(screenshot)
rect = plt.Rectangle((best_x, best_y), w, h, edgecolor='red', facecolor='none', linewidth=2)
ax.add_patch(rect)
plt.title(f"Best Match (Score: {best_score:.3f})")
plt.show()
