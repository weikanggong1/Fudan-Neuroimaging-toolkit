"""表面放置的隔离只读观测器；保存完整目标/试步，不参与生产接受。"""
from __future__ import annotations

from contextlib import AbstractContextManager
from functools import wraps
from pathlib import Path
import hashlib
import json
import numpy as np


def array_digest(array):
    """数组→dtype、shape、C顺序字节SHA；不导出坐标或影像。"""
    value = np.asarray(array)
    return dict(dtype=str(value.dtype), shape=list(value.shape),
                sha256=hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest())


class PlacementObserver(AbstractContextManager):
    """单进程独立benchmark的局部包装器；不适用于线程并发。

    stage_module为当前place_white_preaparc_python或place_pial_python模块；
    directory为新的私有观测目录。进入后包装既有函数，退出恢复原对象，
    异常也恢复。包装函数照常调用原函数，返回原对象；仅复制诊断数据。
    callback按现有trace_callback契约接收step/pass/坐标/诊断；生成完整
    各步npz与目标搜索npz及仅含数值/哈希的events.json。npz不供生产读入。
    算法/JIT/GPU/写出时间含观测开销，不可宣称生产性能提升。
    """

    def __init__(self, *, stage_module, directory: str | Path):
        self.module = stage_module
        self.directory = Path(directory)
        self.originals = {}
        self.events = []
        self.current = {}
        self.border_index = self.cleanup_index = 0

    def _wrap(self, name, recorder):
        original = getattr(self.module, name)
        self.originals[name] = original
        @wraps(original)
        def observed(*args, **kwargs):
            before = recorder(args, kwargs, None, True)
            result = original(*args, **kwargs)
            recorder(args, kwargs, (before, result), False)
            return result
        setattr(self.module, name, observed)

    def __enter__(self):
        self.directory.mkdir(parents=True, exist_ok=False)
        try:
            self._wrap("compute_border_values_first_pass", self._border)
            self._wrap("intensity_gradient", self._intensity)
            self._wrap("unconstrained_step_with_offsets", self._proposal)
            self._wrap("asynchronous_first_step", self._collision)
            self._wrap("repair_intersections", self._cleanup)
            if hasattr(self.module, "pin_medial_wall"):
                self._wrap("pin_medial_wall", self._pin)
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *exception):
        for name, original in self.originals.items():
            setattr(self.module, name, original)
        self.originals.clear()
        return False

    def _save_event(self, kind, arrays, **metadata):
        index = len(self.events)
        filename = f"{index:04d}-{kind}.private.npz"
        np.savez(self.directory / filename, **arrays)
        self.events.append(dict(kind=kind, file=filename,
            arrays={key: array_digest(value) for key, value in arrays.items()}, **metadata))
        path = self.directory / "events.json"
        temporary = path.with_suffix(".pending")
        temporary.write_text(json.dumps(self.events, indent=2)+"\n")
        temporary.replace(path)

    def _border(self, args, kwargs, data, before):
        if before:
            return {name: np.asarray(args[index]).copy() for name, index in
                    (("coordinates", 2), ("normals", 3), ("original", 4),
                     ("ripped", 5), ("previous_values", 6), ("thresholds", 8))}
        inputs, result = data
        arrays = dict(inputs)
        arrays.update(zip(("raw_values", "distances", "means", "target_coordinates", "marked", "used_sigma"), result))
        self._save_event("border", arrays, pass_index=self.border_index,
            sigma=float(kwargs.get("sigma", 2.)), surface=kwargs.get("surface", "white"))
        self.current["ripped"] = inputs["ripped"]
        self.border_index += 1

    def _proposal(self, args, kwargs, data, before):
        if before:
            return dict(start=np.asarray(args[0]).copy(), gradient=np.asarray(args[1]).copy(),
                        ripped=np.asarray(args[2]).copy())
        inputs, result = data
        self.current.update(inputs)
        self.current["proposed"] = result[0].copy()
        self.current["offsets"] = result[1].copy()
        self.current.setdefault("trial_arrays", []).append(dict(inputs,
            proposed=result[0].copy(), offsets=result[1].copy()))

    def _intensity(self, args, kwargs, data, before):
        if before:
            return {name: np.asarray(args[index]).copy() for name, index in
                    (("intensity_start", 1), ("normals", 2), ("ripped", 3),
                     ("target_values", 4), ("target_sigma", 5))}
        inputs, result = data
        self.current.update(inputs)
        self.current["intensity_gradient"] = np.asarray(result).copy()

    def _collision(self, args, kwargs, data, before):
        if before: return None
        _, result = data
        self.current["trial_arrays"][-1]["candidate"] = result[0].copy()

    def _cleanup(self, args, kwargs, data, before):
        if before:
            return dict(before=np.asarray(args[0]).copy(), faces=np.asarray(args[1]).copy(),
                        ripped=np.asarray(args[2]).copy())
        inputs, result = data
        self._save_event("cleanup", dict(inputs, after=result[0]),
                         cleanup_index=self.cleanup_index, diagnostics=result[1])
        self.cleanup_index += 1

    def _pin(self, args, kwargs, data, before):
        if before: return np.asarray(args[0]).copy()
        initial, result = data
        self._save_event("pin", dict(before=initial, after=result, white=args[1], cortex=args[2]))

    def callback(self, step, pass_index, coordinates, diagnostics):
        arrays = {key: value for key, value in self.current.items() if key != "trial_arrays"}
        arrays["accepted"] = np.asarray(coordinates).copy()
        for index, trial in enumerate(self.current.get("trial_arrays", [])):
            for key, value in trial.items(): arrays[f"trial{index}_{key}"] = value
        self._save_event("step", arrays, step=int(step), pass_index=int(pass_index), diagnostics=diagnostics)
        self.current = {}
