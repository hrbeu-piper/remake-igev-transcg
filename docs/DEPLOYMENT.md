# 部署与验收

以下命令在 Linux GPU 服务器上执行。资源根固定为 `/root/autodl-tmp/cvpr`；`/root/autodl-tmp` 必须是真实独立可写数据盘，与系统根目录不同文件系统，写入后保留至少 80 GiB。不要把 Docker 环境目录 `/opt/remake-env` 用宿主目录覆盖，也不要在挂载缺失时创建系统盘替代目录。

## 镜像启动

先从本次交付清单取得镜像 tar 和 SHA-256，并在服务器核验。镜像由官方 `crane` 按固定基础镜像 digest、环境层、源码层组装；本次镜像验收范围为组装结构及层身份，**未使用 Docker daemon 启动，也未验证容器内 CUDA**。

```bash
sha256sum -c SHA256SUMS
docker load -i remake-igev-transcg-20261026-cu128.tar
```

本次镜像名称为 `hrbeu-piper/remake-igev-transcg:20261026-cu128`，加载后核对该 tag。下面先运行默认健康检查，再进入显式部署会话。Docker 与 NVIDIA Container Toolkit 须在目标机器上可用，镜像不会安装宿主 GPU 驱动。

```bash
IMAGE='hrbeu-piper/remake-igev-transcg:20261026-cu128'
docker run --rm "$IMAGE"
docker run --rm --gpus all "$IMAGE" \
  python /opt/remake-igev/scripts/healthcheck.py --require-cuda
docker run --rm -it --gpus all --shm-size=8g \
  --mount type=bind,src=/root/autodl-tmp,dst=/root/autodl-tmp \
  "$IMAGE" bash
```

默认健康检查核对源码清单、固定包版本和 CPU 导入；`--require-cuda` 额外执行小型 CUDA tensor 前反向。这两者均不能代替完整模型资格。镜像默认不下载数据、不安装项目、不开始训练。

若需通过 Dockerfile 重建，先将已核验的 `env-layer.tar.gz` 放在仓库根目录。Dockerfile 默认已固定 NVIDIA CUDA 12.8.2 Ubuntu 22.04 的 linux/amd64 基础镜像 `nvcr.io/nvidia/cuda@sha256:0d3cb6d53388903d4292c575da1576d0c1ef1f54e475892d7ea8d0b38490b4ce`，无需填写占位 digest：

```bash
docker build --platform linux/amd64 \
  -t hrbeu-piper/remake-igev-transcg:20261026-cu128 .
```

环境归档由 `scripts/export_environment.py` 调用固定 `conda-pack 0.8.1` 生成，目标前缀 `/opt/remake-env`，归档根 `opt/remake-env`。本次归档大小为 4,281,868,870 bytes，SHA-256 `a21674cd610a943157b06dc1cd08113f981053fd931fc778109e51276652c2b3`。它需要作为独立交付物取得；只有 Git 源码和依赖列表无法重建这份既有环境快照。

`ADD env-layer.tar.gz /` 展开后已完成前缀替换；不要再次运行 `conda-unpack`，也不要把别的环境归档改名替代。固定快照的导出和镜像层校验不承诺不同驱动／硬件上的数值或性能相同。

没有 Docker daemon 时，可在 Linux 服务器使用实际组装入口 `scripts/build_image_crane.py`：

```bash
python scripts/build_image_crane.py \
  --package-root /path/to/remake-igev-transcg \
  --env-layer /path/to/env-layer.tar.gz \
  --crane /path/to/crane \
  --output-dir /path/on-independent-data-disk/image-output
```

脚本固定校验官方 crane 二进制和环境层 SHA，使用同一个基础镜像 digest，生成 Docker tar、源码 zip、`SHA256SUMS` 与 `image-receipt.json`，并验证完整镜像层。输出目录须位于独立可写数据盘且有至少 30 GiB 可用空间；具体服务器挂载 UUID 由调用端另行核验。该路径验证归档结构和层哈希，不会启动容器，不能作为 Docker 运行或容器内 CUDA 通过的证明。

## 安装源码与准备固定八参考资格

以下在上面的容器会话中执行。`bootstrap_runtime.py` 仅复制清单中的源码/vendor，绑定新 deployment；旧检查点、旧数据回执、旧 CUDA 资格和旧缓存不会被复制。已有不同项目目录时它会停止，请保留原项目并明确处理版本冲突。

```bash
export RES=/root/autodl-tmp/cvpr
export PROJECT="$RES/projects/remake_igev"
export CONTROL="$RES/control/remake-igev"
export PYTHONPATH="$PROJECT/src"
export TMPDIR="$RES/tmp"
BOOT=/opt/remake-igev/scripts/bootstrap_runtime.py

python "$BOOT" install
python "$BOOT" verify
mkdir -p "$TMPDIR"
python "$BOOT" download-da
python "$BOOT" manifest --output "$RES/manifests/reference-only.json"
python "$BOOT" eight-ids --manifest "$RES/manifests/reference-only.json" \
  --output "$CONTROL/eight-ids.json"
```

`manifest` 需要全部场景的原始 metadata 位于 `$RES/datasets/transcg`。它只验证并固定结构及场景划分，不证明完整图像已上传；`eight-ids` 固定取排序后的前 8 个 train 参考，不按文件是否存在或 GT 难度选择。先准备这些参考的 RGB、raw、GT mask 和 GT depth，再显式生成 CUDA 资格缓存：

```bash
python -m remake_igev.single_cache \
  --manifest "$RES/manifests/reference-only.json" \
  --project-root "$PROJECT" --resource-root "$RES" \
  --cache-dir "$RES/cache/single-da" \
  --fingerprint-out "$CONTROL/da-fingerprint.json" \
  --status-out "$CONTROL/eight-cache.json" \
  --sample-ids-json "$CONTROL/eight-ids.json" --diagnostic-only

python "$BOOT" config --stage baseline --run-id sv-baseline-20261026-v1 \
  --manifest "$RES/manifests/reference-only.json" \
  --fingerprint "$CONTROL/da-fingerprint.json" \
  --cache-dir "$RES/cache/single-da" \
  --cache-receipt "$CONTROL/train-val-cache.json" \
  --verification-receipt "$RES/datasets/.verification/transcg-20261026/verified-receipt.json" \
  --output "$CONTROL/baseline-config.json"

python -m remake_igev.single_qualify \
  --config "$CONTROL/baseline-config.json" \
  --resource-root "$RES" --project-root "$PROJECT" \
  --output "$CONTROL/qualifications/baseline.json"
```

八参考缓存只能标记 partial；不能作为 34,191 个 train/val 参考的完整缓存。资格使用 200 步，要求有限更新、相应训练分支非零梯度、规定的 loss 下降及完整恢复；诊断权重不会用于正式初始化。每个阶段均需对应资格。此段不会启动全量训练。

## 新机器验收边界

打包的 `src/verify_autodl_transcg.py` 是原实例的严格验收实现，包含 `connect.nmb2.seetacloud.com:28143`、bzl 数据源、挂载和 SSH 身份等绑定。**它尚不是任意新主机可直接使用的通用验收器。** 迁移前需审核这些固定来源和目标，适配验证实现并冻结新版本、更新源码清单与 deployment，然后重新执行本实例的全树校验；有源码变动的 CUDA 资格也需按新版本重新生成。bootstrap 不会自动改绑定或替换 verifier。

必须具备当前实例的源／目标全树校验结果、零差异证据、完整库存和 SHA 绑定才能生成新的 verified receipt。复制旧实例回执、手写 `complete=true`、以归档下载结束替代校验、取消 80 GiB 预留均不能使验收通过。发布的原 verifier、部署清单和资格记录之间有哈希约束，不能只改一份配置绕过。

完成上述适配和真实验收之后，才执行一次性训练链：

```bash
python -m remake_igev.single_pipeline \
  --resource-root "$RES" --project-root "$PROJECT" \
  --control-dir "$CONTROL" --wait-hours 12
```

链路复核基线资格，等待并校验完整数据回执，执行仅 train 的深度范围统计，生成及逐项校验完整 train/val DA 缓存，然后按 B40 → baseline_continue20 → R5 → F15 执行。每阶段使用 val 选择 best，并保留完整 optimizer、scheduler、随机状态和采样游标。源码变化、空间不足、校验失败或资格失败都会停止并留证，不自动放宽检查或反复重试。查看 `$CONTROL/pipeline-status.json`、`pipeline-events.jsonl` 与 `logs/` 判断阶段；进程存在不等于完成。

## 冻结测试与容量

只有 `baseline_continue` 与 `fusion` 两条分支均完整结束、checkpoint 链及各自 val-best 锁定后，才运行 `single_predict`。按一个模型一次预测和评分，保留实际 80 GiB 余量；不要求两模型预测同时常驻。程序不会自动删除训练缓存或既有预测，空间不足会停止。

```bash
python -m remake_igev.single_predict --help
python -m remake_igev.single_rescore --help
```

正式 `single_predict` 需要同时提供 `--baseline-run`、`--fusion-run`、`--stage`、`--resource-root`、`--project-root`、`--output-root`。逐参考生成 DA 缓存，预测提交并核验后才清理该参考可重建的临时缓存。独立 `single_rescore` 用 `--manifest`、`--prediction-dir`、`--output-dir`、`--resource-root` 重新读取评分来源，核齐全部 23,524 个 test 参考并做 9 遍精确 P95 统计；不能用 `--diagnostic-only` 产生正式成绩。

完成某模型预测和评分后，先把完整产物归档并逐 SHA 验证，再另行授权清理远端派生副本，随后处理另一个模型。报告条件始终保留 `gt_privileged`，不能将 GT-mask 成绩写为自动 mask 部署效果。
