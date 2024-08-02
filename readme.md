## 苗带检测

### 版本
python 3.8.19   
torch 1.12.1  
torchvision 0.13.1   
cuda 11.7.0  
### 环境配置
conda create -n env_name python==3.8  
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch  
pip install requirements.txt  
其他少了那个安装那个  
### 数据集准备
python mytraining_data_example/test.py  
### 训练
python train.py --model pidnet/bisenet  
<p>这是替换DBSCAN聚类算法之后的模块，pidnet再替换之后效果比bisenet要好。</p>
python Endtoend_train.py --model pidnet/bisenet  

### 测试
python evalue.py --model pidnet/bisenet  
<p>这是替换DBSCAN聚类算法之后的模块，pidnet再替换之后效果比bisenet要好。在2080上测试，从图片输入到最终得到苗带参数fps大于30帧。后续将在车道数据集上测试，并优化网络.</p>
python Endtoend_evalue.py --model pidnet/bisenet  

### 权重
bisenet  
best_model_checkpoint_351.pth  
last_model_checkpoint_351.pth  

pidnet  
best_model_pidnet_checkpoint.pth  
last_model_pidnetcheckpoint.pth  

