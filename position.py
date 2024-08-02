import torch

def create_horizontal_position_encoding(height, width, d_model):
    """
    创建仅包含水平位置编码的矩阵，并扩展到指定的维度
    :param height: 图像高度
    :param width: 图像宽度
    :param d_model: 编码的维度
    :return: 扩展到指定维度的水平位置编码矩阵
    """
    # 创建水平位置编码，从0到1递增
    x_position = torch.linspace(0, 1, width).unsqueeze(0)  # (1, width)
    
    # 将水平位置编码扩展到图像的高度
    position_enc = x_position.expand(height, width)  # (height, width)
    
    # 扩展到指定的编码维度
    position_enc = position_enc.unsqueeze(0)  # (1, height, width)
    position_enc = position_enc.expand(d_model, height, width)  # (d_model, height, width)
    
    return position_enc

# 示例
height, width, d_model = 960, 544, 4
position_enc = create_horizontal_position_encoding(height, width, d_model)

print(position_enc.shape)  # 输出形状应为 (d_model, height, width)
print(position_enc)