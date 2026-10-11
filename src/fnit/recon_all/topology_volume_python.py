"""自有数字拓扑修复候选：6/26 simple-point生长，再复用FNIT有序tessellation。

完整不同算法候选，不是FS GA逐位复现，不切生产默认。逐体素原位决策
不并行近似，不以Euler数代替连通性、非流形、自交和形状检查。
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import heapq
import json
from pathlib import Path
import time

import nibabel as nib
import nibabel.freesurfer.io as fsio
import numpy as np
from numba import njit
from scipy import ndimage

from .extract_main_component_python import extract_main_component
from .mris_remove_intersection_python import mark_intersections
from .tessellate_gpu import tessellate_mgh, write_quad_surface
from .topology_preflight_python import topology_counts


_OFFSETS = np.asarray([(x-1,y-1,z-1) for x in range(3) for y in range(3) for z in range(3)], np.int64)
_FACE = np.asarray([4,10,12,14,16,22], np.int64)
_N18 = np.sum(np.abs(_OFFSETS), axis=1) <= 2
_N18[13] = False
_SQUARES = []
for _axis in range(3):
    _other = [a for a in range(3) if a != _axis]
    for _a in range(2):
        for _b in range(2):
            _indices = []
            for _first in range(2):
                for _second in range(2):
                    _xyz = [1, 1, 1]
                    _xyz[_other[0]], _xyz[_other[1]] = _a+_first, _b+_second
                    _indices.append(9*_xyz[0]+3*_xyz[1]+_xyz[2])
            _SQUARES.append(_indices)
_SQUARES = np.asarray(_SQUARES, np.int64)
_CUBES = np.asarray([[9*(x+dx)+3*(y+dy)+z+dz
                      for dx in range(2) for dy in range(2) for dz in range(2)]
                     for x in range(2) for y in range(2) for z in range(2)], np.int64)


@njit(cache=True, fastmath=False)
def _components(local, foreground):
    """T6只数N18中与中心6相邻的分量；T26数N26全部背景分量。"""
    visited = np.zeros(27, np.bool_)
    queue = np.empty(27, np.int64)
    count = 0
    for start_slot in range(6 if foreground else 27):
        start = _FACE[start_slot] if foreground else start_slot
        if start == 13 or visited[start] or local[start] != foreground:
            continue
        count += 1
        if count > 1:
            return count
        first, last = 0, 1
        queue[0], visited[start] = start, True
        while first < last:
            current = queue[first]
            first += 1
            for other in range(27):
                if other == 13 or visited[other] or local[other] != foreground:
                    continue
                if foreground and not _N18[other]:
                    continue
                dx = abs(_OFFSETS[current,0]-_OFFSETS[other,0])
                dy = abs(_OFFSETS[current,1]-_OFFSETS[other,1])
                dz = abs(_OFFSETS[current,2]-_OFFSETS[other,2])
                linked = dx+dy+dz == 1 if foreground else max(dx,dy,dz) == 1
                if linked:
                    visited[other], queue[last] = True, other
                    last += 1
    return count


@njit(cache=True, fastmath=False)
def _well_composed_addition(local):
    """拒绝中心新增引起的2×2棋盘格和2×2×2互补对角临界配置。"""
    local[13] = True
    for ids in _SQUARES:
        if local[ids[0]] == local[ids[3]] and local[ids[1]] == local[ids[2]] and local[ids[0]] != local[ids[1]]:
            local[13] = False
            return False
    for ids in _CUBES:
        count = 0
        for i in range(8):
            count += int(local[ids[i]])
        if count == 2 or count == 6:
            first, second = -1, -1
            for i in range(8):
                if local[ids[i]] == (count == 2):
                    if first == -1:
                        first = i
                    else:
                        second = i
            if (first ^ second) == 7:
                local[13] = False
                return False
    local[13] = False
    return True


@njit(cache=True, fastmath=False)
def _grow(target, priority, seed):
    shape = target.shape
    result = np.zeros(shape, np.bool_)
    state = np.zeros(shape, np.uint8)
    flat_result, flat_state = result.ravel(), state.ravel()
    flat_target, flat_priority = target.ravel(), priority.ravel()
    flat_result[seed], flat_state[seed] = True, 3
    accepted = np.empty(int(target.sum()), np.int64)
    accepted[0] = seed
    total = 1
    tests = rejected_topology = rejected_composition = maximum_heap = 0
    heap = [(np.float64(0), np.int64(seed))]
    heapq.heappop(heap)
    sx, sy, sz = shape
    x, y, z = seed//(sy*sz), seed//sz % sy, seed % sz
    for slot in _FACE:
        dx,dy,dz = _OFFSETS[slot]
        index = ((x+dx)*sy+y+dy)*sz+z+dz
        if flat_target[index]:
            heapq.heappush(heap, (-flat_priority[index], index))
            flat_state[index] = 1
    local = np.empty(27, np.bool_)
    while heap:
        maximum_heap = max(maximum_heap, len(heap))
        _, index = heapq.heappop(heap)
        flat_state[index] = 2
        x, y, z = index//(sy*sz), index//sz % sy, index % sz
        for slot in range(27):
            dx,dy,dz = _OFFSETS[slot]
            local[slot] = result[x+dx,y+dy,z+dz]
        tests += 1
        if _components(local, True) != 1 or _components(local, False) != 1:
            rejected_topology += 1
            continue
        if not _well_composed_addition(local):
            rejected_composition += 1
            continue
        flat_result[index], flat_state[index] = True, 3
        accepted[total] = index
        total += 1
        # 每个接受改变26邻域的简单性；重新评估阻塞点，不截断候选。
        for slot in range(27):
            if slot == 13:
                continue
            dx,dy,dz = _OFFSETS[slot]
            nx,ny,nz = x+dx,y+dy,z+dz
            if nx < 1 or nx >= sx-1 or ny < 1 or ny >= sy-1 or nz < 1 or nz >= sz-1:
                continue
            other = (nx*sy+ny)*sz+nz
            if flat_target[other] and flat_state[other] != 1 and flat_state[other] != 3:
                heapq.heappush(heap, (-flat_priority[other], other))
                flat_state[other] = 1
    return result, accepted[:total], np.asarray([tests,rejected_topology,rejected_composition,maximum_heap], np.int64)


@njit(cache=True, fastmath=False)
def _nonmanifold_vertex_links(faces, nvertices):
    """闭合边流形之外，检查每个顶点link是否只有一条连通环。"""
    counts=np.zeros(nvertices,np.int64)
    for face in faces:
        for vertex in face:
            counts[vertex]+=1
    offsets=np.empty(nvertices+1,np.int64)
    offsets[0]=0
    for vertex in range(nvertices):
        offsets[vertex+1]=offsets[vertex]+counts[vertex]
    cursor=offsets[:-1].copy()
    incident=np.empty(3*len(faces),np.int64)
    for number,face in enumerate(faces):
        for vertex in face:
            incident[cursor[vertex]]=number
            cursor[vertex]+=1
    defective=0
    for vertex in range(nvertices):
        degree=counts[vertex]
        if degree==0:
            defective+=1
            continue
        link=np.empty((degree,2),np.int64)
        for local in range(degree):
            column=0
            for other in faces[incident[offsets[vertex]+local]]:
                if other!=vertex:
                    link[local,column]=other
                    column+=1
        visited=np.zeros(degree,np.bool_)
        stack=np.empty(degree,np.int64)
        visited[0],stack[0]=True,0
        first,last=0,1
        while first<last:
            current=stack[first]
            first+=1
            for other in range(degree):
                if visited[other]:
                    continue
                if (link[current,0]==link[other,0] or link[current,0]==link[other,1]
                        or link[current,1]==link[other,0] or link[current,1]==link[other,1]):
                    visited[other],stack[last]=True,other
                    last+=1
        if last!=degree:
            defective+=1
    return defective


def correct_binary_topology(*, target: np.ndarray, spacing_mm: tuple[float,float,float] = (1.,1.,1.)) -> tuple[np.ndarray,dict]:
    """完整优先队列修复候选，返回同网格bool掩膜与全部删除/决策诊断。

    target为3D bool白质目标；spacing_mm默认三轴1mm，须正且有限。要求
    目标不接触图像边缘；先选最大6连通分量（同体积时先出现者），在EDT
    最大的内部种子开始、按内部深度优先和线性索引稳定同分次序加入simple
    voxel。每次使用当前已接受状态，保留6前景/26背景数字拓扑；另排除
    体素边界的临界面/角配置。终止于完整队列为空，不缩短迭代或用球替代。
    输出是输入目标子集；会切断handle，不填目标外体素，不等价于FS GA。
    报告包含分量、种子、删除数量/Dice、全部接受顺序SHA和测试/拒绝次数；
    空输入/非法空间/边界接触抛ValueError。此阶段尚不能判整体指标等效。
    """
    started = time.perf_counter()
    mask = np.asarray(target, dtype=bool)
    spacing = np.asarray(spacing_mm, np.float64)
    if mask.ndim != 3 or min(mask.shape) < 3 or spacing.shape != (3,) or not np.isfinite(spacing).all() or (spacing<=0).any():
        raise ValueError("expected 3D mask and three positive finite spacing_mm")
    if not mask.any():
        raise ValueError("empty topology target")
    if any(np.take(mask, [0,mask.shape[axis]-1], axis=axis).any() for axis in range(3)):
        raise ValueError("topology target touches image boundary")
    labels, components = ndimage.label(mask, structure=ndimage.generate_binary_structure(3,1))
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    selected = int(sizes.argmax())
    connected = np.ascontiguousarray(labels == selected)
    priority = ndimage.distance_transform_edt(connected, sampling=spacing)
    seed = int(priority.argmax())
    setup_done = time.perf_counter()
    repaired, order, counters = _grow(connected, np.ascontiguousarray(priority), seed)
    count = int(repaired.sum())
    report = {"algorithm": "ordered_6_26_simple_addition_with_well_composed_boundary_candidate",
              "input_foreground_voxels": int(mask.sum()), "input_components_6": int(components),
              "largest_component_voxels": int(sizes[selected]), "output_foreground_voxels": count,
              "removed_foreground_voxels": int(mask.sum())-count,
              "foreground_dice_against_target": 2*count/(int(mask.sum())+count),
              "seed_voxel_index_xyz": [int(index) for index in np.unravel_index(seed,mask.shape)],
              "spacing_mm": spacing.tolist(), "accepted_order_sha256": hashlib.sha256(order.astype('<i8').tobytes()).hexdigest(),
              "simple_point_tests": int(counters[0]), "topology_rejections": int(counters[1]),
              "well_composed_rejections": int(counters[2]), "maximum_heap_items": int(counters[3]),
              "setup_seconds": setup_done-started, "growth_seconds_including_first_JIT": time.perf_counter()-setup_done,
              "total_seconds": time.perf_counter()-started, "overall_equivalence": "not_assessed"}
    return repaired, report


def run_volume_topology(*, input_volume: str | Path, value: int, output_volume: str | Path,
                        output_surface: str | Path, device: str = "cpu",
                        source_surface: str | Path | None = None,
                        output_defect_labels: str | Path | None = None,
                        report_path: str | Path | None = None) -> dict:
    """同一pretess uint8 MGZ→修复标签与orig.premesh候选，完整结构QC。

    四路径/标签必填，value为1..255目标标签；device默认cpu，只控制复用
    FNIT tessellation设备，拓扑决策始终Numba CPU。volume保留scanner RAS
    affine/原网格/uint8及所有非目标标签；surface为原几何surface RAS/mm。
    输出路径不得覆盖输入。产生私有quad中间文件以复用完整有序tessellate
    与主分量提取；返回所有路径/SHA、分量、Euler、边、相交、边界体素
    距离和包含读取/JIT/写出的墙钟。闭合/单分量/流形/正面积/零相交任一
    失败抛RuntimeError，不伪造拓扑成功；下游sphere及指标仍须整例验证。
    source_surface/output_defect_labels默认None，提供时必须成对：从原
    orig.nofix映射实际删除区并写同序整数图；新算法定义见对应函数。
    """
    started = time.perf_counter()
    input_volume, output_volume, output_surface = map(Path,(input_volume,output_volume,output_surface))
    if len({p.resolve() for p in (input_volume,output_volume,output_surface)}) != 3:
        raise ValueError("input/output paths must differ")
    if (source_surface is None)!=(output_defect_labels is None):
        raise ValueError('source_surface and output_defect_labels must be provided together')
    if not isinstance(value,(int,np.integer)) or isinstance(value,bool) or not 1 <= value <= 255:
        raise ValueError("value must be integer in 1..255")
    image = nib.load(str(input_volume))
    data = np.asanyarray(image.dataobj)
    if data.dtype != np.uint8 or not hasattr(image.header,"get_vox2ras_tkr"):
        raise ValueError("requires uint8 MGH/MGZ with declared surface RAS geometry")
    repaired, report = correct_binary_topology(target=data==value, spacing_mm=tuple(image.header.get_zooms()[:3]))
    corrected = data.copy()
    corrected[(data==value)&~repaired] = 0
    output_volume.parent.mkdir(parents=True,exist_ok=True)
    output_surface.parent.mkdir(parents=True,exist_ok=True)
    nib.save(nib.MGHImage(corrected,image.affine,header=image.header.copy()),str(output_volume))
    vertices,quads = tessellate_mgh(input_path=output_volume,value=int(value),device=device)
    quad_path = output_surface.with_name(output_surface.name+'.topology-candidate.quad')
    write_quad_surface(path=quad_path,vertices=vertices,quads=quads,image=image,input_path=output_volume)
    components = extract_main_component(quad_path,output_surface)
    xyz,faces = fsio.read_geometry(str(output_surface))
    counts = topology_counts(faces,len(xyz))
    nonmanifold_vertices=int(_nonmanifold_vertex_links(np.ascontiguousarray(faces,np.int32),len(xyz)))
    directed_edges=np.concatenate((faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]))
    edge_sign=np.where(directed_edges[:,0]<directed_edges[:,1],1,-1)
    undirected=np.sort(directed_edges,axis=1)
    _,inverse=np.unique(undirected,axis=0,return_inverse=True)
    orientation_conflicts=int(np.count_nonzero(np.bincount(inverse,weights=edge_sign)))
    _,intersections = mark_intersections(xyz,faces)
    area = np.linalg.norm(np.cross(xyz[faces[:,1]]-xyz[faces[:,0]],xyz[faces[:,2]]-xyz[faces[:,0]]),axis=1)/2
    edge = ndimage.generate_binary_structure(3,1)
    before_boundary = (data==value)&~ndimage.binary_erosion(data==value,structure=edge)
    after_boundary = repaired&~ndimage.binary_erosion(repaired,structure=edge)
    distances_to_before = ndimage.distance_transform_edt(~before_boundary,sampling=image.header.get_zooms()[:3])[after_boundary]
    distances_to_after = ndimage.distance_transform_edt(~after_boundary,sampling=image.header.get_zooms()[:3])[before_boundary]
    quality = (components==1 and counts.euler==2 and counts.boundary_edges==0 and counts.nonmanifold_edges==0
               and nonmanifold_vertices==0 and orientation_conflicts==0
               and intersections==0 and (area>0).all())
    report.update(input_volume=str(input_volume),output_volume=str(output_volume),output_surface=str(output_surface),
                  value=int(value),device=device,coordinate_units='surface_RAS_mm',volume_dtype='uint8',
                  input_sha256=hashlib.sha256(input_volume.read_bytes()).hexdigest(),
                  output_volume_sha256=hashlib.sha256(output_volume.read_bytes()).hexdigest(),
                  output_surface_sha256=hashlib.sha256(output_surface.read_bytes()).hexdigest(),
                  quad_intermediate=str(quad_path),
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),mesh=asdict(counts),
                  raw_surface_components=int(components),geometric_intersecting_faces=int(intersections),
                  nonmanifold_vertices=nonmanifold_vertices,edge_orientation_conflicts=orientation_conflicts,
                  zero_area_faces=int((area==0).sum()),mesh_quality_status='passed_structural_candidate' if quality else 'failed',
                  boundary_voxel_distance_mm={'method':'bidirectional_boundary_voxel_EDT_not_triangle_surface_distance',
                      'before_to_after_max':float(distances_to_after.max()),'before_to_after_p99':float(np.percentile(distances_to_after,99)),
                      'after_to_before_max':float(distances_to_before.max()),'after_to_before_p99':float(np.percentile(distances_to_before,99))},
                  total_file_API_seconds=time.perf_counter()-started,native_program_called=False)
    if source_surface is not None:
        report['defect_labels']=write_volume_topology_defects(input_volume=input_volume,
            corrected_volume=output_volume,source_surface=source_surface,value=value,
            output_defect_labels=output_defect_labels)
        report['total_file_API_seconds']=time.perf_counter()-started
    if report_path is not None:
        Path(report_path).parent.mkdir(parents=True,exist_ok=True)
        Path(report_path).write_text(json.dumps(report,indent=2,ensure_ascii=False,default=int)+'\n')
    if not quality:
        raise RuntimeError('digital topology candidate failed complete structural mesh quality; see preserved report')
    return report


def write_volume_topology_defects(*, input_volume: str | Path, corrected_volume: str | Path,
                                  source_surface: str | Path, value: int,
                                  output_defect_labels: str | Path,
                                  report_path: str | Path | None = None) -> dict:
    """数字拓扑删除区→原orig.nofix同序整数morph，不借用原GA标签。

    两个uint8 MGZ须同shape/affine，原volume目标标签value为1..255；修复
    volume只能删除该标签而不能改变其他值。source_surface必须是原voxel
    tessellation的orig.nofix三角面，其surface RAS/mm及体积几何头声明
    同一空间。将删除区6连通分量编号映射到每个源顶点邻接的8个体素，
    多区域接触取最小正编号，未接触为0；返回(N,)整值float32 morph。
    定义是新算法切断区，不是FS GA defect编号逐位复现。报告包含SHA、
    区域/顶点计数、空间、算法和含读取/映射/写入的时间。原文件保留；
    非角点/无几何/输入不匹配/非法标签抛异常，不写占位输出。既有
    defects_to_volume可按相同原顶点序投射此图，不替代新网格质量检查。
    """
    from .place_surface_geometry import surface_ras_to_voxel
    started=time.perf_counter()
    input_volume,corrected_volume,source_surface,output_defect_labels=map(
        Path,(input_volume,corrected_volume,source_surface,output_defect_labels))
    if len({p.resolve() for p in (input_volume,corrected_volume,source_surface,output_defect_labels)})!=4:
        raise ValueError('input, corrected volume, source surface and output paths must differ')
    if not isinstance(value,(int,np.integer)) or isinstance(value,bool) or not 1<=value<=255:
        raise ValueError('value must be integer in 1..255')
    before_image,after_image=nib.load(str(input_volume)),nib.load(str(corrected_volume))
    before,after=np.asanyarray(before_image.dataobj),np.asanyarray(after_image.dataobj)
    if (before.dtype!=np.uint8 or after.dtype!=np.uint8 or before.shape!=after.shape
            or not np.array_equal(before_image.affine,after_image.affine)):
        raise ValueError('expected same-grid uint8 original and corrected MGZ')
    if np.any((before!=after)&((before!=value)|(after!=0))):
        raise ValueError('corrected volume must only remove the selected target label')
    removed=(before==value)&(after==0)
    labels,count=ndimage.label(removed,structure=ndimage.generate_binary_structure(3,1))
    xyz,faces,metadata=fsio.read_geometry(str(source_surface),read_metadata=True)
    if not metadata:
        raise ValueError('source orig.nofix requires declared volume geometry')
    matrix=surface_ras_to_voxel(before_image.header,metadata)
    voxel=xyz@np.asarray(matrix[:3,:3],np.float64).T+np.asarray(matrix[:3,3],np.float64)
    low=np.floor(voxel).astype(np.int64)
    if not np.isfinite(voxel).all() or np.max(np.abs(voxel-low-.5))>1e-4:
        raise ValueError('source surface must retain original voxel-corner tessellation positions')
    assigned=np.zeros(len(xyz),np.int32)
    shape=np.asarray(before.shape)
    for dx in range(2):
        for dy in range(2):
            for dz in range(2):
                index=low+np.asarray([dx,dy,dz])
                valid=np.all((index>=0)&(index<shape),axis=1)
                touching=np.zeros(len(xyz),np.int32)
                touching[valid]=labels[tuple(index[valid].T)]
                selected=(touching>0)&((assigned==0)|(touching<assigned))
                assigned[selected]=touching[selected]
    output_defect_labels.parent.mkdir(parents=True,exist_ok=True)
    fsio.write_morph_data(str(output_defect_labels),assigned.astype(np.float32),fnum=len(faces))
    report={'scope':'digital topology cut-region diagnostic on original orig.nofix vertices',
            'definition':'minimum positive 6-connected deleted-region label in eight corner-adjacent voxels',
            'original_GA_label_equivalence':'different_algorithm_not_assessed',
            'input_volume':str(input_volume),'corrected_volume':str(corrected_volume),
            'source_surface':str(source_surface),'output_defect_labels':str(output_defect_labels),
            'input_sha256':{name:hashlib.sha256(p.read_bytes()).hexdigest() for name,p in
                            [('original_volume',input_volume),('corrected_volume',corrected_volume),('source_surface',source_surface)]},
            'output_sha256':hashlib.sha256(output_defect_labels.read_bytes()).hexdigest(),
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'value':int(value),'deleted_voxels':int(removed.sum()),'deleted_regions_6':int(count),
            'vertices':len(xyz),'faces':len(faces),'nonzero_label_vertices':int(np.count_nonzero(assigned)),
            'mapped_positive_labels':int(np.unique(assigned[assigned>0]).size),
            'coordinate_units':'surface_RAS_mm','output_shape':[len(xyz)],'dtype':'float32_exact_integers',
            'vertex_order':'unchanged original orig.nofix','native_program_called':False,
            'total_seconds_including_io':time.perf_counter()-started}
    if report_path is not None:
        Path(report_path).parent.mkdir(parents=True,exist_ok=True)
        Path(report_path).write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-volume',type=Path,required=True)
    parser.add_argument('--value',type=int,required=True)
    parser.add_argument('--output-volume',type=Path,required=True)
    parser.add_argument('--output-surface',type=Path,required=True)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--source-surface',type=Path)
    parser.add_argument('--output-defect-labels',type=Path)
    parser.add_argument('--report-path',type=Path,required=True)
    args=parser.parse_args()
    run_volume_topology(input_volume=args.input_volume,value=args.value,output_volume=args.output_volume,
                        output_surface=args.output_surface,device=args.device,source_surface=args.source_surface,
                        output_defect_labels=args.output_defect_labels,report_path=args.report_path)


if __name__=='__main__':
    main()
