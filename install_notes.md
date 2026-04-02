# NetKet 3.21 GPU 安装记录

## 当前目标

- 在远程 LSF 集群上使用 `NetKet 3.21`
- 使用 `GPU JAX`
- 保持 `editable install`，因为本地已经修改了 `netket` 源码

## 已确认环境

- 宿主机系统 `glibc = 2.17`
- 已有可用的 `conda` 和 `Python 3.11` 环境
- 集群提供 `module` 系统
- 集群类型为 `LSF`

## 已遇到的问题

### 1. 直接安装新版 `jax[cuda12]` 失败

执行：

```bash
python -m pip install -U "jax[cuda12]" netket
```

出现的问题：

- `pip` 在解析依赖时回退到了源码构建
- `ml_dtypes` 的构建依赖拉到了 `numpy` 源码包
- 构建过程中使用的 C 编译器仍然是系统默认的 `gcc 4.8.5`
- 最终报错：`NumPy requires GCC >= 9.3`

这说明当前宿主机环境无法稳定安装 `NetKet 3.21` 对应的新版 JAX GPU 依赖栈。

### 2. 误加载了 `glibc/2.36-gcc12.1.0`

最初希望通过 `module load glibc/2.36-gcc12.1.0` 提升底层运行时版本。

但实际结果是：

- `which gcc` 等外部命令直接 `Segmentation fault`
- 原因不是 `gcc` 本身，而是该模块修改了 `LD_LIBRARY_PATH`
- 导致系统程序启动时混用了不兼容的 `glibc`

### 3. `glibc` module 本身存在打包问题

进一步检查后发现：

```bash
readelf -d /fs00/software/glibc/2.36-gcc12.1.0/lib/ld-linux-x86-64.so.2 | egrep 'RPATH|RUNPATH'
```

输出为：

```text
0x000000000000000f (RPATH)  Library rpath: [/fs00/software/gcc/12.1.0/lib64]
```

随后尝试直接调用该 loader 运行 Python：

```bash
/fs00/software/glibc/2.36-gcc12.1.0/lib/ld-linux-x86-64.so.2 ...
```

报错：

```text
Inconsistency detected by ld.so: ./get-dynamic-info.h: 134: elf_get_dynamic_info: Assertion `info[DT_RPATH] == NULL' failed!
```

这表明这套 `glibc` module 至少不适合直接作为用户态运行时来承载 Python/JAX 环境。

## 当前判断

- 问题不只是 `gcc` 版本老
- 真正卡住的是：`NetKet 3.21` 依赖的新版 `JAX GPU` 栈对宿主机 `glibc 2.17` 不友好
- 集群提供的 `glibc/2.36-gcc12.1.0` 模块也不能直接用于修复这个问题
- 因此，不再继续尝试在宿主机裸环境中强行安装 `NetKet 3.21 + GPU JAX`

## 拟采用方案

### 使用 Apptainer/Singularity 容器

目标方案：

- 使用带新用户态环境的容器，例如 `python:3.11-bullseye`
- 在容器内部创建独立 Python 虚拟环境
- 在容器内部安装 `jax[cuda12]`
- 将宿主机上的 `netket` 源码目录 bind 到容器内
- 在容器内对源码执行：

```bash
python -m pip install -e .
```

这样可以同时满足：

- 使用 `NetKet 3.21`
- 使用 GPU 版 JAX
- 保留本地源码修改并进行 `editable install`
- 绕开宿主机 `glibc 2.17` 和损坏的 `glibc module`

## 后续执行要点

1. 检查集群上可用的是 `apptainer` 还是 `singularity`
2. 拉取基础容器镜像，例如 `docker://python:3.11-bullseye`
3. 在容器内创建 venv
4. 安装 `jax[cuda12]`
5. 对当前 `netket` 源码执行 `pip install -e .`
6. 在 LSF 的 GPU 作业中通过 `apptainer exec --nv` 运行

## 备注

- 宿主机已有的 `conda` 环境不再作为最终运行环境
- 以后真正运行 `NetKet 3.21 + GPU JAX` 时，以容器内环境为准
- 宿主机主要保留源码、数据和作业脚本

## 容器安装完整流程

以下流程假设当前目录就是 `netket` 源码根目录。

### 1. 检查容器命令

```bash
command -v apptainer || command -v singularity
```

如果输出的是 `apptainer`，后文中的容器命令直接使用 `apptainer`。
如果输出的是 `singularity`，则将后文中的 `apptainer` 替换为 `singularity`。

### 2. 准备镜像和容器内虚拟环境目录

```bash
export CTR=apptainer
export IMG=$HOME/containers/python311-bullseye.sif
export NK_SRC=$PWD
export NK_VENV=$HOME/venvs/netket321-container

mkdir -p "$HOME/containers"
mkdir -p "$HOME/venvs"
```

如果集群只有 `singularity`，则改成：

```bash
export CTR=singularity
```

### 3. 拉取基础镜像

```bash
$CTR pull "$IMG" docker://python:3.11-bullseye
```

### 4. 清理宿主机环境变量污染

目标是避免宿主机旧的 `conda`、`LD_LIBRARY_PATH`、`PYTHONPATH` 污染容器。

```bash
module purge
conda deactivate 2>/dev/null || true
unset PYTHONPATH
unset LD_LIBRARY_PATH
unset CONDA_PREFIX
unset CONDA_DEFAULT_ENV
```

### 5. 在容器内创建 venv 并安装 JAX / NetKet

```bash
$CTR exec --cleanenv -B "$NK_SRC":/workspace/netket "$IMG" bash -lc "
python -m venv '$NK_VENV'
source '$NK_VENV/bin/activate'
python -m pip install -U pip setuptools wheel
python -m pip install -U 'jax[cuda12]'
cd /workspace/netket
python -m pip install -e . --no-build-isolation
python -m pip check
"
```

说明：

- `-B "$NK_SRC":/workspace/netket` 用于把宿主机源码目录绑定到容器中
- `pip install -e .` 保证后续修改宿主机源码后，容器内也会直接生效
- `--no-build-isolation` 可以减少某些构建时重复拉依赖的问题

## LSF 交互式测试

先申请一个带 GPU 的交互节点，具体队列名可能需要按集群实际情况调整。

常见形式之一：

```bash
bsub -Is -q gpu -gpu "num=1" bash
```

有些集群也可能需要：

```bash
bsub -Is -q gpu -R "select[gpu]" -gpu "num=1" bash
```

进入计算节点后，测试容器内 GPU JAX 是否正常：

```bash
export CTR=apptainer
export IMG=$HOME/containers/python311-bullseye.sif
export NK_SRC=/你的/netket/源码绝对路径
export NK_VENV=$HOME/venvs/netket321-container

$CTR exec --cleanenv --nv -B "$NK_SRC":/workspace/netket "$IMG" bash -lc "
source '$NK_VENV/bin/activate'
python - <<'PY'
import jax
import netket as nk
print('JAX:', jax.__version__)
print('NetKet:', nk.__version__)
print('Devices:', jax.devices())
PY
"
```

如果输出中能看到 `GpuDevice(...)`，说明容器方案基本可用。

## LSF 批处理脚本模板

可以保存成 `run_netket_lsf.sh`：

```bash
#!/bin/bash
#BSUB -J netket
#BSUB -q gpu
#BSUB -n 4
#BSUB -gpu "num=4"
#BSUB -o netket.%J.out
#BSUB -e netket.%J.err

export CTR=apptainer
export IMG=$HOME/containers/python311-bullseye.sif
export NK_SRC=/你的/netket/源码绝对路径
export NK_VENV=$HOME/venvs/netket321-container

$CTR exec --cleanenv --nv -B "$NK_SRC":/workspace/netket "$IMG" bash -lc "
source '$NK_VENV/bin/activate'
export NETKET_EXPERIMENTAL_SHARDING=1
cd /workspace/netket
python your_script.py
"
```

提交方式：

```bash
bsub < run_netket_lsf.sh
```

## 多 GPU 备注

- 单节点多 GPU 时，优先先跑通容器 + `jax.devices()`
- 再在作业里开启 `NETKET_EXPERIMENTAL_SHARDING=1`
- 如果需要限制可见 GPU，可在宿主机先设置：

```bash
export APPTAINERENV_CUDA_VISIBLE_DEVICES=0,1,2,3
```

然后再执行：

```bash
$CTR exec --cleanenv --nv ...
```

## 实施建议

建议按下面顺序执行：

1. 先确认 `apptainer` 或 `singularity` 可用
2. 拉取 `python:3.11-bullseye` 容器
3. 在容器内安装 `jax[cuda12]`
4. 在容器内对当前 `netket` 源码执行 `pip install -e .`
5. 用 LSF 交互式 GPU 作业测试 `jax.devices()`
6. 最后再提交正式的批处理作业
