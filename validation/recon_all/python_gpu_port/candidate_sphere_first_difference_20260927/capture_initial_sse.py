"""Capture the first native sphere SSE arrays and stop before optimization.

Requires SPHERE_CAPTURE_DIR and the pinned FreeSurfer 8.2 mris_sphere binary.
This GDB script supports the Python 2.7 embedded in gpucw1 GDB 7.6.1.
"""

import hashlib
import json
import os
import re
import struct
import sys
import time

import gdb


def digest(path):
    result = hashlib.sha256()
    with open(path, "rb") as stream:
        while True:
            block = stream.read(1 << 20)
            if not block:
                break
            result.update(block)
    return result.hexdigest()


class FirstSSE(gdb.Breakpoint):
    def __init__(self):
        gdb.Breakpoint.__init__(self, "*0x4ac8f0")
        self.hits = 0

    def stop(self):
        self.hits += 1
        if self.hits != 1:
            return False
        started = time.time()
        out = os.environ["SPHERE_CAPTURE_DIR"]
        if not os.path.isdir(out):
            os.makedirs(out)
        inferior = gdb.selected_inferior()
        register_text = gdb.execute("info registers rbp", to_string=True)
        mris = int(re.search(r"0x[0-9a-fA-F]+", register_text).group(0), 16)
        header = bytes(inferior.read_memory(mris, 48))
        nvertices = struct.unpack_from("<i", header, 4)[0]
        topology_ptr = struct.unpack_from("<Q", header, 32)[0]
        vertices_ptr = struct.unpack_from("<Q", header, 40)[0]
        if not 100000 < nvertices < 200000:
            raise RuntimeError("unexpected vertex count {}".format(nvertices))
        topology = bytes(inferior.read_memory(topology_ptr, nvertices * 48))
        vertices = bytes(inferior.read_memory(vertices_ptr, nvertices * 464))
        with open(os.path.join(out, "native_topology_structs.bin"), "wb") as stream:
            stream.write(topology)
        with open(os.path.join(out, "native_vertex_structs.bin"), "wb") as stream:
            stream.write(vertices)

        offsets = bytearray((nvertices + 1) * 8)
        xyz = bytearray(nvertices * 12)
        total = 0
        with open(os.path.join(out, "native_target.bin"), "wb") as targets:
            with open(os.path.join(out, "native_current.bin"), "wb") as currents:
                for vno in xrange(nvertices):
                    vertex_offset = 464 * vno
                    xyz[12 * vno:12 * (vno + 1)] = vertices[vertex_offset + 24:vertex_offset + 36]
                    count = struct.unpack_from("<h", topology, 48 * vno + 38)[0]
                    current_ptr, target_ptr = struct.unpack_from("<QQ", vertices, vertex_offset)
                    if not 0 < count < 1024 or not current_ptr or not target_ptr:
                        raise RuntimeError("invalid metric row {}: {}, {}, {}".format(
                            vno, count, current_ptr, target_ptr))
                    currents.write(inferior.read_memory(current_ptr, count * 4))
                    targets.write(inferior.read_memory(target_ptr, count * 4))
                    total += count
                    struct.pack_into("<Q", offsets, (vno + 1) * 8, total)
        with open(os.path.join(out, "native_offsets.bin"), "wb") as stream:
            stream.write(offsets)
        with open(os.path.join(out, "native_xyz.bin"), "wb") as stream:
            stream.write(xyz)

        expected = (3, 1, 47, 94827, 42, 51, 50, 4, 7, 8)
        neighbor_ptr_offset = None
        for candidate in range(0, 48, 8):
            pointer = struct.unpack_from("<Q", topology, candidate)[0]
            if pointer < 4096:
                continue
            try:
                first = struct.unpack("<10i", inferior.read_memory(pointer, 40))
            except gdb.MemoryError:
                continue
            if first == expected:
                neighbor_ptr_offset = candidate
                break
        if neighbor_ptr_offset is not None:
            with open(os.path.join(out, "native_neighbor_ids.bin"), "wb") as neighbors:
                for vno in xrange(nvertices):
                    count = struct.unpack_from("<h", topology, 48 * vno + 38)[0]
                    pointer = struct.unpack_from(
                        "<Q", topology, 48 * vno + neighbor_ptr_offset)[0]
                    neighbors.write(inferior.read_memory(pointer, count * 4))

        layout = {"neighbor_ptr_offset": neighbor_ptr_offset}
        extra_names = []
        for field in ("VERTEX", "VERTEX_TOPOLOGY"):
            try:
                layout[field + "_size"] = int(gdb.parse_and_eval("sizeof({})".format(field)))
            except gdb.error as error:
                layout[field + "_size_error"] = str(error)
        try:
            flag_offset = int(gdb.parse_and_eval(
                "(unsigned long)&(((VERTEX*)0)->ripflag)"))
            flag_size = int(gdb.parse_and_eval("sizeof(((VERTEX*)0)->ripflag)"))
            if not 0 <= flag_offset < 464 or flag_size not in (1, 2, 4):
                raise ValueError("unexpected ripflag layout")
            layout["vertex_ripflag_offset"] = flag_offset
            layout["vertex_ripflag_size"] = flag_size
            flags = "".join(vertices[464 * v + flag_offset:
                                     464 * v + flag_offset + flag_size]
                            for v in xrange(nvertices))
            with open(os.path.join(out, "native_vertex_ripflags.bin"), "wb") as stream:
                stream.write(flags)
            extra_names.append("native_vertex_ripflags.bin")
        except (gdb.error, ValueError) as error:
            layout["vertex_ripflag_error"] = str(error)
        try:
            pointer = int(gdb.parse_and_eval("parms->vsmoothness"))
            layout["vsmoothness_pointer"] = hex(pointer)
            if pointer:
                element_size = int(gdb.parse_and_eval("sizeof(parms->vsmoothness[0])"))
                layout["vsmoothness_element_size"] = element_size
                with open(os.path.join(out, "native_vsmoothness.bin"), "wb") as stream:
                    stream.write(inferior.read_memory(pointer, nvertices * element_size))
                extra_names.append("native_vsmoothness.bin")
        except (gdb.error, gdb.MemoryError) as error:
            layout["vsmoothness_pointer_error"] = str(error)

        names = ("native_topology_structs.bin", "native_vertex_structs.bin",
                 "native_target.bin", "native_current.bin", "native_offsets.bin",
                 "native_xyz.bin")
        if neighbor_ptr_offset is not None:
            extra_names.append("native_neighbor_ids.bin")
        names += tuple(extra_names)
        result = {
            "breakpoint": "0x4ac8f0",
            "logSSE_hit": self.hits,
            "vertex_count": nvertices,
            "distance_count": total,
            "capture_seconds": time.time() - started,
            "layout": layout,
            "sha256": dict((name, digest(os.path.join(out, name))) for name in names),
        }
        with open(os.path.join(out, "native_capture.json"), "w") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")
        print("NATIVE_FIRST_SSE_CAPTURE " + json.dumps(result))
        sys.stdout.flush()
        return True


gdb.execute("set pagination off")
gdb.execute("set confirm off")
FirstSSE()
gdb.execute("run")
