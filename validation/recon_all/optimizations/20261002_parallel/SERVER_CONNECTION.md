# 五会话复用服务器连接

本轮五个会话共用协调者已经认证的 OpenSSH ControlMaster。连接只负责执行和传输；
成功连接不能作为算法、精度或整例性能通过的证据。

## 入口与检查

运行期配置和包装器保存在协调者的私有目录，不提交密码、密钥、许可证或认证内容。
每个会话读取同一配置，使用同一个 socket。先执行 `ssh -O check`，再通过该连接
核对 `hostname`、`whoami` 和 `pwd`。本轮五个会话均已确认目标为 gpucw1、账号
gongwk；各自的连接结果记录在私有 `task_N.connection.json`。

```bash
# 将下面的路径设为协调者给出的私有包装器目录；不要另建认证连接。
fnit_connection_directory=/path/to/private/coordination

# 检查已有主连接后，在服务器执行只读身份核对。
"$fnit_connection_directory/ssh_gpucw1.sh" 'hostname; whoami; pwd'

# 传输各任务自己的源码归档，传输后在服务器核验其 SHA-256。
"$fnit_connection_directory/scp_gpucw1.sh" \
  /path/to/task_source.tar.gz gpucw1:/path/to/task/source.tar.gz
```

SSH 包装器使用 `ssh -S SOCKET`；SCP 使用 `scp -o ControlPath=SOCKET`。
`scp -S` 指定的是 SSH 程序路径，不能传入 socket。两个入口都使用 BatchMode，
连接失效时明确失败，避免各任务各自停在密码提示。子会话不得执行 `ssh -O exit`、
删除 socket 或终止共享主进程。失效时由协调者检查和恢复主连接。

新会话的本地执行目录和 shell 也应指向实际 WSL 路径与 `/bin/bash`。
`Failed to create unified exec process` 属于本地执行入口问题，应先核对 shell/cwd；
尚未执行 SSH 的错误不能标为服务器认证失败。

## 计时任务排队

开发、只读检查和不计时 CPU 测试可以独立进行；同机性能测量及 GPU 测试取得
同一把服务器本地 `/tmp` 文件锁。当前存储的 GPFS 路径不支持所需的 flock，
此前 ENOLCK 尝试均记为未执行，改用本地锁后重新运行。

```bash
# 所有任务使用协调者给出的同一个锁路径，不能分别创建一把锁。
fnit_benchmark_lock=/tmp/fnit-shared-benchmark.lock

# 在服务器取得锁后运行本任务的一个完整计时命令，结束即释放。
flock "$fnit_benchmark_lock" /path/to/conda/bin/python /path/to/benchmark.py

# 只读检查持锁者与等待者；等待锁不能记入算法耗时。
lslocks --output PID,MODE,BLOCKER,PATH
```

等待锁记为 `waiting_shared_lock`，连接失败记为 `connection_failed`，实际子进程
启动后才记为 `running`。不要为缩短排队时间取消其他任务或并发污染配对计时。

## 本轮验证范围

五个会话分别完成身份核对；共用 SCP 的小文件 SHA-256 传输检查也通过。
本说明解决会话之间的连接复用，不宣称安装隔离、完整原始 T1 运行或整体提速已通过。
