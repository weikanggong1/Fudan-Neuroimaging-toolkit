# NVIDIA RM ioctl 状态独立诊断

## 用途与边界

`nvidia_ioctl_status_probe.c` 是 Linux x86_64 的独立 LD_PRELOAD 验证工具，用于本轮新 scalar 子进程。它记录 NVIDIA RM 固定 ioctl 头的返回状态，帮助区分 syscall 成功和 RM 内部失败。它不调用 CUDA，不修改 driver 参数，不链接进 FNIT 生产程序，不修改冻结 816e5610 源码。仅由协调者在共享 benchmark 锁内运行自己的新诊断进程。

此前 v4 两个子进程几乎同时在 context 创建边界返回 CUDA code 2，而 strace 的 NVIDIA ioctl 全部返回 0。此工具用于补充 strace 无法显示的固定头 status；CUDA code、NV_STATUS、Linux errno 是三个不同的编号空间，不能直接等同。它本身不能确定 host 页分配或 GPU 容量是哪一种原因。

## 固定 ABI 与数据最小化

依据 NVIDIA 官方 **535.216.03** tag 的头文件及 `escape.c` 参数大小检查，独立描述下列结构。仅支持 read/write、type `F`、明确 number/size 的 ioctl：

| number | 固定头 | 大小 bytes | 标识整数偏移 bytes | status 偏移 bytes |
| --- | --- | ---: | ---: | ---: |
| 0x2a RM_CONTROL | NVOS54 | 32 | cmd: 8 | 28 |
| 0x2b RM_ALLOC | NVOS21 | 32 | class: 12 | 28 |
| 0x2b RM_ALLOC | NVOS64 | 48 | class: 12 | 40 |

编译时 static assert 检查布局。32 位 ioctl 编号允许 x86_64 的零扩展或符号扩展；其他高位模式不解码。其他 ioctl 原参数直接转发。未支持的 UVM、RM_FREE、transfer 包装等不会记录或沿指针继续解码。

仅在 libc ioctl 返回非负值之后，使用 `process_vm_readv` **读取本进程**最多 48 bytes 固定头到局部缓冲区。不会直接解引用用户指针，不读取嵌套参数、rights、句柄对应对象或任意内存。不输出地址、buffer 内容、环境清单或凭据。固定头复制失败时 `header_copied=false`，标识/status 为 null；不会因此改变原 ioctl 结果。若调用者同时修改该头，日志只是复制时的观察，不能保证与内核返回瞬间一致。

## 构建与调用（由协调者部署）

现场已确认既有 Conda 环境的 `x86_64-conda-linux-gnu-gcc` 可用，版本 11.4.0。源码不需要 CUDA headers 或新增依赖。以下是构建建议，**本子任务没有在服务器构建或启动 GPU**：

```bash
# 在独立诊断源码目录构建，产物保留在该诊断目录。
FNIT_DIAG_PYTHON_ENV=/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929/fnit_main_env
"${FNIT_DIAG_PYTHON_ENV}/bin/x86_64-conda-linux-gnu-gcc" \
  -std=c11 -O2 -Wall -Wextra -Werror -shared -fPIC \
  nvidia_ioctl_status_probe.c -ldl -o nvidia_ioctl_status_probe.so
sha256sum nvidia_ioctl_status_probe.c nvidia_ioctl_status_probe.so

# 仅附加到协调者已有的新 scalar 子进程命令；先持有 /tmp/fnit-shared-benchmark.lock。
# 下列占位参数须替换为已审查的 scalar 诊断，不是生产入口。
FNIT_NV_IOCTL_TRACE_PATH=/absolute/new_diagnostic_run/child_ioctl.jsonl \
LD_PRELOAD=/absolute/new_diagnostic_source/nvidia_ioctl_status_probe.so \
"${FNIT_DIAG_PYTHON_ENV}/bin/python" /absolute/existing_scalar_diagnostic.py
```

输入只有环境字段 `FNIT_NV_IOCTL_TRACE_PATH`（输出文件路径）。父目录必须预先存在。首次加载时以 `O_APPEND|O_CLOEXEC|O_NOFOLLOW`、新文件权限 0600 打开；已有文件的权限不会修改，协调者应使用独立新目录。打开失败时继续原 ioctl、没有日志。不要给生产队列或其他人的进程注入。原命令如已有 LD_PRELOAD，应先审核加载顺序。

## 输出与实现限制

JSONL 每行包含 `schema`、`pid`、`tid`、`realtime_ns`、`monotonic_ns`、`fd`、`request`、`number`、`size`、`kind`、`cmd` 或 `class`、`status`、`header_copied`、`ret`、`errno`。时间戳是在调用完成后的日志时间。状态为无符号整数，未加入未经查证的状态名。`errno` 始终为调用返回后的原值；ret 成功时 errno 可能是以前留下的数值，不能解释为此次失败。

转发使用 `dlsym(RTLD_NEXT,"ioctl")`；加载期间递归调用以同参数 raw syscall 转发。线程局部递归保护避免日志递归，没有跨线程 mutex 或 fork 后遗留的锁。每条记录一次 append write，不重试失败或部分写入，因此有 IO 错误时可能缺记录，不能把缺记录解释为没有失败。不会拦截直接 `syscall(SYS_ioctl,...)` 或隐藏/内部 libc 绑定，覆盖程度须与同轮 strace 对照。工具增加少量 host 记录开销，应做带/不带 preload 的配对；不能称为 benchmark。

此实现限定 glibc/Linux x86_64 register-word 第三参数 ABI；不会把未给 payload 的请求当指针读。运行时加载应使用 fresh exec，未验证运行中 dlopen、卸载或其他 libc/架构。

## 已做验证与待验证

2026-10-04 本地 GCC `-std=c11 -Wall -Wextra -Werror` 语法及 shared 构建通过。无 GPU 的 LD_PRELOAD 替身链测试通过：NVOS54 status=81、NVOS21/NVOS64 status=0，正确读取 cmd/class；ret=0、原 errno=117 完全保留；size=31 的请求不产生记录。替身测试仅验证包装/布局逻辑，不是 NVIDIA driver 实测，也不是真实 benchmark。临时测试源码及产物在 `/tmp/fnit-ioctl-probe-localtest`。

待协调者在既有 Conda compiler 构建，记录产物 SHA，执行自己的 scalar 对照并确认 trace 覆盖、process_vm_readv 是否受策略限制、实际 NV_STATUS 与 CUDA 返回码的对应。生产初始化与算法仍没有修改。

## 官方依据

- [535.216.03 nv-ioctl-numbers.h：magic F](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/535.216.03/kernel-open/common/inc/nv-ioctl-numbers.h)
- [535.216.03 nv_escape.h：RM_CONTROL/RM_ALLOC 编号](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/535.216.03/src/nvidia/arch/nvalloc/unix/include/nv_escape.h#L30)
- [535.216.03 nvos.h：NVOS21/NVOS64](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/535.216.03/src/common/sdk/nvidia/inc/nvos.h#L453)
- [535.216.03 nvos.h：NVOS54](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/535.216.03/src/common/sdk/nvidia/inc/nvos.h#L2170)
- [535.216.03 escape.c：RM_ALLOC 大小检查](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/535.216.03/src/nvidia/arch/nvalloc/unix/src/escape.c#L363)
- [535.216.03 escape.c：RM_CONTROL 大小检查](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/535.216.03/src/nvidia/arch/nvalloc/unix/src/escape.c#L744)

上述官方源码仅现场读取到 /tmp 供核查，未复制驱动实现到 FNIT。没有使用任何系统脑影像软件二进制。
