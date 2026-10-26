# ReMake + IGEV++ 式单路深度补全

这是 `hrbeu-piper/remake-igev-transcg` 的部署仓库，现为公开仓库，封装当前 AutoDL 训练实现。模型输入为同一参考图像的 RGB、原始深度、有效位、冻结 Depth Anything V2 Small 相对深度和透明区域 mask；新增多范围候选评分、聚合与 GRU 细化。它不读取第二张 RGB、相机内参或相机间位姿，候选评分体不代表双目匹配体。

**当前证据边界：原 AutoDL 实例的基线已通过固定 8 个训练参考、200 步 CUDA 资格检查与完整状态恢复；尚无完整训练／测试精度结论。** 打包镜像的层与哈希检查也不代表 Docker 中 CUDA 已运行通过。新机器必须重新生成数据验收、CUDA 资格和运行记录。

## 固定内容

- 运行源码快照：`78d64525597bed485bef8ef8e3b5f139a0224943`。全部 56 个 `remake_igev` Python 模块及顶层数据 verifier 由 [runtime-manifest.json](runtime-manifest.json) 逐文件 SHA-256 绑定；vendor 同样冻结。
- 单路入口：`single_manifest`、`single_cache`、`single_qualify`、`single_pipeline`、`single_train`、`single_predict`、`single_rescore`。历史双目模块保留供既有公共导入使用，部署方案不调用旧双目训练入口。
- ReMake vendor 仅保留本次运行实现及辅助组件子集，不分发未使用的其他模型；保留文件的数值源码保持冻结版本不变。
- 环境快照：Python 3.12.3、PyTorch 2.8.0+cu128、torchvision 0.23.0+cu128、NumPy 2.3.2、opencv-python-headless 4.11.0.86。环境以固定前缀 `/opt/remake-env` 导出，不是重新解算依赖。
- 数据、权重、单目缓存、检查点、旧机器回执、环境归档和镜像归档均不进入 Git。DA V2 Small 通过显式下载命令获得并核验 SHA。

## 实验协议

| 内容 | 固定设置 |
|---|---|
| 数据划分 | 按场景隔离：68 train / 10 val / 52 test；参考数 29,861 / 4,330 / 23,524 |
| 输入网格与单位 | 480 × 640；D435 深度单位 0.001 m，L515 为 0.00025 m |
| mask 条件 | `gt_privileged`：使用 GT mask；不是无需 GT 的部署结果 |
| 基线 B | 40 epochs，从头训练 ReMake |
| 同预算比较 | 同一个 B 的 val-best → ReMake 续训 20 epochs；或 R 预热 5 epochs → F 联合细化 15 epochs |
| 数值配置 | FP32，物理 batch 1，累积 8，identity 增强；matmul/cuDNN TF32、cuDNN benchmark 均关闭；cuDNN deterministic=false |
| 正式 test | 两条比较分支完整完成、各自 val-best 锁定后才能读取；逐模型推理、独立精确评分 |

GT 深度只进入监督和评分，不进入模型 inputs；不会为搜索范围或指标裁切 GT。训练范围统计仅使用 train。DA 先输出原生图像尺寸，再按既有数值顺序变换至训练网格并缓存；缓存来源和 backend 都纳入指纹。

## 部署入口

先读 [部署说明](docs/DEPLOYMENT.md)。服务器必须具有 NVIDIA GPU 运行条件、独立可写数据盘 `/root/autodl-tmp`，全部项目资源放在 `/root/autodl-tmp/cvpr`。每次写入后必须保留至少 **80 GiB**；数据盘容量合计还要容纳数据、约 40 GiB 单目缓存、检查点和当前一个模型的预测，不可只按镜像大小规划。

镜像包含 `/opt/remake-igev` 源码包和 `/opt/remake-env` 环境，默认仅运行健康检查。部署时执行：

```bash
python /opt/remake-igev/scripts/healthcheck.py
python /opt/remake-igev/scripts/bootstrap_runtime.py install
python /opt/remake-igev/scripts/bootstrap_runtime.py verify
```

`install` 校验包清单并把源码/vendor 复制到数据盘项目目录，生成新的源码 deployment 记录；**不会启动训练，也不会声明数据或 CUDA 已验收**。已有不同内容的项目会拒绝覆盖。

当前 `src/verify_autodl_transcg.py` 绑定原 AutoDL 端点、bzl 数据源和主机证据。迁移到另一台服务器时，需要审核并适配这些绑定，重新冻结部署版本和生成完整校验回执；复制旧 `verified-receipt.json` 或关闭检查均不能完成迁移。完整训练在此前会主动停止。详见 [新机器验收边界](docs/DEPLOYMENT.md#新机器验收边界)。

## 镜像与许可证

[Dockerfile](Dockerfile) 对应实际环境层导入路线：固定 NVIDIA CUDA 12.8.2 Ubuntu 22.04 基础镜像 digest，加 `conda-pack 0.8.1` 导出的 `env-layer.tar.gz`，再加冻结源码/vendor。环境导出使用 `dest_prefix=/opt/remake-env`、`arcroot=opt/remake-env`，无需启动时执行 `conda-unpack`。重建需要单独取得并核验这份环境归档；本仓库不提供由 `pip freeze` 重新求解依赖即可复现的承诺。

已导出的环境层为 4,281,868,870 bytes，SHA-256 `a21674cd610a943157b06dc1cd08113f981053fd931fc778109e51276652c2b3`，保留 223 个名称含 license/copying 的归档成员；这不是完整许可证合规审计计数。本次无 Docker daemon，镜像通过官方 `crane` 组装为可加载的 Docker tar；尚未完成容器启动和容器内 CUDA 测试。最终镜像 SHA 及验证结果以交付清单为准。

本仓库未为自有代码授予开源许可。ReMake、Depth Anything V2 及环境中各依赖保持各自许可证和署名，见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
