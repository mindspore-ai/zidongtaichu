# omni-perception-pretrainer

#### 模型介绍
OPT（Omni-Perception Pre-Trainer）是全场景感知预训练模型的简称，中文名字叫紫东太初，是中科院自动化和华为联合研发的多模态预训练模型，本仓是紫东太初十亿参数级别模型的MindSpore版本代码，包含预训练模型及多个下游任务模型。

#### 环境安装
参考微调组件环境安装（AICC场景）。



#### 镜像准备

参考 `code/docker/README.md` 准备并上传镜像。

备注：当使用其他模型的时候，请同步使用其配套的DockerFile。


#### 模型准备


按照如下文件结构准备训练资源

1. [下载opt模型代码 adapt_tk分支](https://gitee.com/mindspore/omni-perception-pretrainer/tree/adapt_tk/)：

    ```shell
    git clone https://gitee.com/mindspore/omni-perception-pretrainer.git -b adapt_tk
    ```

2. 依据如下目录树下载数据集：   

   [下载COCO caption数据集](https://pan.baidu.com/s/1ECN5JXlRPQsBS8O763Y8pA)（提取码84me），在`{opt模型根目录}/dataset`目录下解压；

   [下载COCO图片数据训练集](http://images.cocodataset.org/zips/train2014.zip)，将所有图片解压至`{opt模型根目录}/dataset/data/train/img/mscoco/train2014/`路径；

   [下载COCO图片数据测试集](http://images.cocodataset.org/zips/val2014.zip)，将所有图片解压至`{opt模型根目录}/dataset/data/train/img/mscoco/val2014/`路径；

   准备任意张以`.jpg`或`.png`为后缀的图片文件，放置在`{opt模型根目录}/dataset/data_infer/`目录下推理数据集。

3. [下载预训练模型文件](https://opt-release.obs.cn-central-221.ovaijisuan.com:443/model/OPT_1-38_136.ckpt)（`OPT_1-38_136.ckpt`）存放至`{opt模型根目录}/pretrained_model`路径。

4. 将云端训练涉及的**应用配置文件**`{opt模型根目录}/code/model_configs/app_config_*.yaml`中路径替换为实际obs路径与镜像路径。

5. （可选）将任务类型对应的模型配置文件`{opt模型根目录}/omni-perception-pretrainer/code/model_configs/model_config_*.yaml`，中的参数替换为实际用户所需参数，也可直接使用示例文件。

6. 准备完成后将`omni-perception-pretrainer`文件夹及其包含文件上传至obs。

    


#### 功能体验

使用微调组件功能前需注册微调组件，运行如下命令，交互输入认证信息：

```shell
fm registry  # 依次输入registry type 1，以及对应的ak，sk，endpoint
```

##### 模型微调

```shell
fm finetune --scenario modelarts --app_config obs://HwAiUser/omni-perception-pretrainer/code/fm_configs/caption/app_config_finetune.yaml --model_config_path obs://HwAiUser/code/fm_configs/caption/model_config_finetune.yaml
```

##### 模型评估

```shell
fm evaluate --scenario modelarts --app_config obs://HwAiUser/omni-perception-pretrainer/code/fm_configs/caption/app_config_evaluate.yaml --model_config_path obs://HwAiUser/code/fm_configs/caption/model_config_evaluate.yaml
```

##### 模型推理

```shell
fm infer --scenario modelarts --app_config obs://HwAiUser/omni-perception-pretrainer/code/fm_configs/caption/app_config_infer.yaml --model_config_path obs://HwAiUser/code/fm_configs/caption/model_config_infer.yaml
```

##### 查看状态

- 查看任务运行状态

```shell
fm job-status --job_id ***  # ***为job_id，任务拉起成功后生成
```


任务结束后，可在任务对应的`app_config_*.yaml`中指定的`output_path`下查看任务输出结果；在指定的`log_path`下查看任务输出日志， 更多功能接口参数详解请参考微调组件文档 。
