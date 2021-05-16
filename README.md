# 2021 Kesci 水下目标检测算法赛  <font color=red>**A榜 mAP 0.569**</font><br /> <font color=red>**B榜 mAP 0.568**</font><br /> 

## 比赛地址：[Kesci 水下目标检测](https://www.heywhale.com/home/competition/605ab78821e3f6003b56a7d8/)

## 整体思路
   + detection algorithm: Detectors Cascade R-CNN 
   + backbone: ResNet50 + RFPN
   + data process:
       1. 去除训练集中u开头的仅含扇贝类的数据(u开头图片扇贝标注噪声大，与测试集分布不一致)
   + data aug:
       1. 多尺度训练 短边704-1216尺度随机
       2. 随机左右翻转
       3. 随机运动模糊
       4. AutoAugment v1
       5. 考虑推理时间复杂度，使用尺度较小导致小目标漏检，因此anchor scale 使用4
   + tta:
       1. 三尺度 
       2. 左右翻转
   + post process: soft nms & max_per_img 300
   
## 实验记录

ATSS在这个数据上不太work，但是还是作为了一个baseline

加~~删除线~~为不work的实验， 粗体为当次实验控制变量变化的内容

|模型|数据预处理|数据增强|anchor策略|TTA|后处理|testA线上分数|
| ------------- |------------- | ------------- | ------------- | ------------- |------------- |------------- |
|cascade dcn r50| 加入无目标图 |多尺度训练（1500-2000）+ 左右翻转|max_iou_assign anchor_scale=8|None|nms|0.52444688|
|cascade dcn r50| 加入无目标图 |多尺度训练（1500-2000）+ 左右翻转| <b>~~ATSS~~</b> anchor_scale=8|None|nms|0.5212299|
|cascade dcn r50| 加入无目标图 |多尺度训练 （1500-2000）+ 左右翻转 + <b>autoaug V1</b>|ATSS anchor_scale=8|None|nms|0.53048|
|cascade dcn r50| 加入无目标图 |多尺度训练 （1500-2000）+ 左右翻转 + <b>label smooth</b>|ATSS anchor_scale=8|None|nms|0.52353581|
|cascade dcn r50| 加入无目标图 |多尺度训练 （1500-2000）+ 左右翻转 + <b>运动模糊</b> |ATSS anchor_scale=8|None|nms|0.52745559|
|cascade dcn r50| 加入无目标图 |多尺度训练 （1500-2000）+ 左右翻转 + <b>~~retinex~~</b> |ATSS anchor_scale=8|None|nms|0.27|
|cascade dcn r50| 加入无目标图 |多尺度训练 （1500-2000）+ 左右翻转 + <b>~~mixup~~</b> |ATSS anchor_scale=8|None|nms|0.51952|
|<b>cascade dcn r101</b>| 加入无目标图 |多尺度训练 （1500-2000）+ 左右翻转 + autoaug V1 + label smooth + 运动模糊|max_iou_assign anchor_scale=8|None|nms|0.53326282|
|<b>Detectors 50</b>| 加入无目标图 |多尺度训练 （1500-2000）+ 左右翻转 |max_iou_assign anchor_scale=8|None|nms|0.53825292|
|Detectors 50| 加入无目标图 |多尺度训练（1500-2000） + <b>左右翻转 + autoaug V1 + label smooth + 运动模糊</b>|max_iou_assign anchor_scale=8|None|nms|0.54643025|
|Detectors 50| 加入无目标图 |多尺度训练（1500-2000） + 左右翻转 + autoaug V1 + label smooth + 运动模糊|max_iou_assign anchor_scale=8|<b>3尺度+左右翻转</b>|nms|0.55482894|
|Detectors 50| 加入无目标图 |多尺度训练（1500-2000） + 左右翻转 + autoaug V1 + label smooth + 运动模糊|max_iou_assign anchor_scale=8|3尺度+左右翻转|<b>soft-nms</b>|0.56624721|
|Detectors 50| <b>去除无目标图</b> |多尺度训练（1500-2000） + 左右翻转 + autoaug V1 + label smooth + 运动模糊|max_iou_assign anchor_scale=8|3尺度+左右翻转|soft-nms|0.56884578|
|Detectors 50| 去除无目标图 |<b>多尺度训练（704-1216）</b> + 左右翻转 + autoaug V1 + label smooth + 运动模糊|max_iou_assign anchor_scale=8|3尺度+左右翻转|soft-nms|0.56453794|
|Detectors 50| 去除无目标图 |多尺度训练（704-1216） + 左右翻转 + <b>~~autoaug V0~~</b> + label smooth + 运动模糊|max_iou_assign anchor_scale=8|3尺度+左右翻转|soft-nms|0.56103191|
|Detectors 50| 去除无目标图 |多尺度训练（704-1216） + 左右翻转 + autoaug V1 + label smooth + 运动模糊|max_iou_assign <b>anchor_scale=4</b>|3尺度+左右翻转|soft-nms|0.5673893|
|<b>cascade resnest 101</b>| 去除无目标图 |多尺度训练（704-1216） + 左右翻转 + autoaug V1 + label smooth + 运动模糊|max_iou_assign anchor_scale=4|3尺度+左右翻转|soft-nms|0.53262004|
|Detectors 50| 去除无目标图 + <b>过滤u开头扇贝类</b> |多尺度训练（704-1216） + 左右翻转 + autoaug V1 + label smooth + 运动模糊|max_iou_assign anchor_scale=4|3尺度+左右翻转|soft-nms|0.56864865|
|<b>cascade x101 64x4d</b>| 去除无目标图 + 过滤u开头扇贝类 |多尺度训练（704-1216） + 左右翻转 + autoaug V1 + label smooth + 运动模糊|max_iou_assign anchor_scale=4|3尺度+左右翻转|soft-nms|0.5557|
|Detectors 50| 去除无目标图 + 过滤u开头扇贝类 |多尺度训练（704-1216） + 左右翻转 + autoaug V1 + label smooth + 运动模糊 + <b>~~instance boost~~</b>|max_iou_assign anchor_scale=4|3尺度+左右翻转|soft-nms|0.55293613|
|Detectors 50| 去除无目标图 + 过滤u开头扇贝类 |多尺度训练（704-1216） + 左右翻转 + label smooth + 运动模糊 + <b>~~去除autoaug~~</b> |max_iou_assign anchor_scale=4|3尺度+左右翻转|soft-nms|0.55283078|
|Detectors 50| 去除无目标图 + 过滤u开头扇贝类 |多尺度训练（704-1216） + 左右翻转 + autoaug V1 + label smooth + 运动模糊|max_iou_assign anchor_scale=4|3尺度+左右翻转|soft-nms + <b>max_per_img 300</b>|0.56927166|

## 代码环境及依赖

+ OS: Ubuntu16.10
+ GPU: 3090 * 8
+ python: python3.7
+ nvidia 依赖:
   - cuda: 11.1
   - cudnn: 8
   - nvidia driver version: 460.32.03
+ deeplearning 框架: pytorch1.7.1
+ 基于mmdetection其他依赖请参考requirement/*.txt （mmcv-full==1.1.5）
+ 显卡数量不太重要，大家依据自身显卡数量倍数调整学习率大小即可

## 训练数据准备

- **相应文件夹创建准备**

  - 在代码根目录下新建data文件夹，或者依据自身情况建立软链接
  - 进入data文件夹,创建文件夹:
  
     annotations

     pretrained

     results

     submit

  - 将官方提供的训练和测试数据解压到data目录中，产生：
    
    train

    test-A-image
    
    test-B-image
    
    
- **label文件格式转换**

  - 官方提供的是VOC格式的xml类型label文件，个人习惯使用COCO格式用于训练，所以进行格式转换
  
  - 使用 tools/process_data/xmltococo_filter.py 将label文件转换为COCO格式，新的label文件 train_filter_old_scallop.json 会保存在 data/train/annotations 目录下 （这一步会同时进行扇贝类的数据清洗）

  - 为了方便测试，我们对test数据也生成一个伪标签文件,运行 tools/process_data/generate_testA_json.py 生成 testA.json, 伪标签文件会保存在data/train/annotations 目录下 （testB同理）

  - 总体运行内容：

    - python tools/process_data/xmltococo_filter.py

    - python tools/process_data/generate_testA_json.py
    
    - python tools/process_data/generate_testB_json.py

- **预训练模型下载**
  - 下载mmdetection官方开源的DetectoRS的COCO预训练模型[detectors_htc_r50_1x_coco-329b1453.pth](http://download.openmmlab.com/mmdetection/v2.0/detectors/detectors_htc_r50_1x_coco/detectors_htc_r50_1x_coco-329b1453.pth)并放置于 data/pretrained 目录下
## 依赖安装及编译


- **依赖安装编译**

   1. 创建并激活虚拟环境
        conda create -n underwater python=3.7 -y
        conda activate underwater

   2. 安装 pytorch
        conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 -c pytorch
   
   3. 安装其他依赖
        pip install -r requirements.txt
   
   4. 安装 mmcv-full (版本为1.1.5)
      按照mmdetection官方说明安装(需要对应cuda及torch版本)
   
   5. 编译 mmdetection：
        python setup.py develop
   

## 模型训练及预测
    
   - **训练**

	1. 运行：
	
		chmod +x tools/dist_train.sh

        ./tools/dist_train.sh configs/underwater/optics.py 8 --no-validate
        
        (上面的8是我的gpu数量，请自行修改)

   	2. 训练过程日志文件及最终权重文件均保存在 work_dirs/optics 目录下

   - **预测**

    1. 运行:
    
        chmod +x tools/dist_test.sh

        ./tools/dist_test.sh configs/underwater/optics.py work_dirs/optics/latest.pth 8 --format-only --eval-options "jsonfile_prefix=./results/testB"

        (上面的8是我的gpu数量，请自行修改)

    2. 预测结果文件会保存在 ./results 目录下 (testB.bbox.json)

    3. 转化mmd预测结果为提交csv格式文件：
       
       python tools/process_data/json2submit.py
       
       最终符合官方要求格式的提交文件 testB.csv 位于 submit目录下
    
## 模型文件final-353579b9.pth下载链接
   [Google云盘](https://drive.google.com/file/d/18xS3F8msfTTED6-IwODlvFNFWs_fU7fz/view?usp=sharing)
   
## Reference
 + [2020 baseline by rill | 斩风](https://github.com/zhengye1995/underwater-object-detection)
 + [2020 Underwater_detection by Wakinguup](https://github.com/Wakinguup/Underwater_detection)
 

## Contact

    author：rill | 斩风

    email：18813124313@163.com
