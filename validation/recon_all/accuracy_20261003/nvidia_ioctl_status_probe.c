/* Independent Linux x86_64 diagnostic; never link into FNIT production.
 * ABI facts: NVIDIA open-gpu-kernel-modules tag 535.216.03, nvos.h,
 * nv_escape.h and escape.c. These are independently described fixed headers;
 * no vendor implementation or pointed-to allocation payload is copied.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <sys/uio.h>
#include <time.h>
#include <unistd.h>
#if !defined(__linux__) || !defined(__x86_64__)
#error "This diagnostic supports only the verified Linux x86_64 ioctl ABI"
#endif

/* uint64_t fields stand for opaque addresses: never dereferenced or logged. */
struct control_header {
    uint32_t client, object, command, flags;
    uint64_t opaque_params;
    uint32_t params_size, status;
};
struct allocation21_header {
    uint32_t root, parent, object, class_id;
    uint64_t opaque_params;
    uint32_t params_size, status;
};
struct allocation64_header {
    uint32_t root, parent, object, class_id;
    uint64_t opaque_params, opaque_rights;
    uint32_t params_size, flags, status;
};
_Static_assert(sizeof(struct control_header) == 32, "NVOS54 size");
_Static_assert(offsetof(struct control_header, command) == 8, "NVOS54 cmd");
_Static_assert(offsetof(struct control_header, status) == 28, "NVOS54 status");
_Static_assert(sizeof(struct allocation21_header) == 32, "NVOS21 size");
_Static_assert(offsetof(struct allocation21_header, status) == 28, "NVOS21 status");
_Static_assert(sizeof(struct allocation64_header) == 48, "NVOS64 size");
_Static_assert(offsetof(struct allocation64_header, status) == 40, "NVOS64 status");
_Static_assert(offsetof(struct allocation64_header, class_id) == 12, "RM class");

static int (*next_ioctl)(int, unsigned long, ...);
static int trace_fd = -1;
static __thread int inside_probe;

__attribute__((constructor)) static void initialize_probe(void)
{
    int saved_errno = errno;
    inside_probe = 1;
    /* Recursive ioctl during loader resolution uses the equivalent raw syscall. */
    void *symbol = dlsym(RTLD_NEXT, "ioctl");
    memcpy(&next_ioctl, &symbol, sizeof(next_ioctl));
    const char *path = getenv("FNIT_NV_IOCTL_TRACE_PATH");
    if (path && *path)
        trace_fd = open(path, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC | O_NOFOLLOW, 0600);
    inside_probe = 0;
    errno = saved_errno;
}

int ioctl(int fd, unsigned long request, ...)
{
    /* Linux x86_64 libc ioctl's third argument is one register-sized word,
     * including ignored words on requests with no payload. Do not interpret it
     * unless the exact type/direction/number/size whitelist below matches. */
    va_list arguments;
    va_start(arguments, request);
    unsigned long argument = va_arg(arguments, unsigned long);
    va_end(arguments);
    int result = next_ioctl ? next_ioctl(fd, request, argument)
                            : (int)syscall(SYS_ioctl, fd, request, argument);
    int ioctl_errno = errno;
    unsigned number = _IOC_NR(request), size = _IOC_SIZE(request);
    /* ioctl command is 32 bits; x86_64 callers may sign-extend constants. */
    uint32_t command32 = (uint32_t)request;
    int canonical = request == (unsigned long)command32 ||
                    request == (unsigned long)(long)(int32_t)command32;
    int supported = canonical && command32 == _IOC(_IOC_READ | _IOC_WRITE, 'F', number, size) &&
                    _IOC_TYPE(request) == 'F' &&
                    _IOC_DIR(request) == (_IOC_READ | _IOC_WRITE) &&
                    ((number == 0x2a && size == 32) ||
                     (number == 0x2b && (size == 32 || size == 48)));
    if (!inside_probe && trace_fd >= 0 && supported) {
        inside_probe = 1;
        unsigned char header[48] = {0};
        int copied = 0;
        uint32_t identifier = 0, status = 0;
        /* Only after successful kernel return. Self process_vm_readv avoids
         * crashing on an unmapped/raced caller buffer; never follow pointers. */
        if (result >= 0) {
            struct iovec local = {header, size};
            struct iovec remote = {(void *)argument, size};
            ssize_t count = syscall(SYS_process_vm_readv, getpid(), &local, 1,
                                    &remote, 1, 0);
            if (count == (ssize_t)size) {
                memcpy(&identifier, header + (number == 0x2a ? 8 : 12), 4);
                memcpy(&status, header + (size == 48 ? 40 : 28), 4);
                copied = 1;
            }
        }
        struct timespec realtime = {0}, monotonic = {0};
        clock_gettime(CLOCK_REALTIME, &realtime);
        clock_gettime(CLOCK_MONOTONIC, &monotonic);
        char status_text[24], identifier_text[24], line[640];
        if (copied) {
            snprintf(status_text, sizeof(status_text), "%u", status);
            snprintf(identifier_text, sizeof(identifier_text), "%u", identifier);
        } else {
            strcpy(status_text, "null");
            strcpy(identifier_text, "null");
        }
        int length = snprintf(line, sizeof(line),
            "{\"schema\":1,\"pid\":%ld,\"tid\":%ld,\"realtime_ns\":%lld,"
            "\"monotonic_ns\":%lld,\"fd\":%d,\"request\":%lu,\"number\":%u,"
            "\"size\":%u,\"kind\":\"%s\",\"%s\":%s,\"status\":%s,"
            "\"header_copied\":%s,\"ret\":%d,\"errno\":%d}\n",
            (long)getpid(), (long)syscall(SYS_gettid),
            (long long)realtime.tv_sec * 1000000000LL + realtime.tv_nsec,
            (long long)monotonic.tv_sec * 1000000000LL + monotonic.tv_nsec,
            fd, request, number, size, number == 0x2a ? "RM_CONTROL" : "RM_ALLOC",
            number == 0x2a ? "cmd" : "class", identifier_text, status_text,
            copied ? "true" : "false", result, ioctl_errno);
        /* One O_APPEND write per record, no userspace mutex: no fork lock or
         * recursive logger deadlock. A failed/partial write is not retried. */
        if (length > 0 && (size_t)length < sizeof(line))
            (void)syscall(SYS_write, trace_fd, line, (size_t)length);
        inside_probe = 0;
    }
    errno = ioctl_errno;
    return result;
}
