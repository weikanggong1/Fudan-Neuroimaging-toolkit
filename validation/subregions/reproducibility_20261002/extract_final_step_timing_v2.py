"""Collect all available actual API/solver timers and official timers; CPU only.

Timer fields from different implementations describe their recorded scope.
No time is inferred from progress timestamps and no intermediate-stage Dice is
fabricated when the official run did not save a comparable label image.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
import re

from build_repeatability_manifest import read_json,sha256
from finalize_reproducibility import write_json

TIMER_PATTERNS=[("preprocessing",re.compile(r"^Preprocessing took (\d+) seconds$")),
                ("atlas_alignment",re.compile(r"^Initial atlas alignment took (\d+) seconds$")),
                ("synthetic_prepare_and_fit",re.compile(r"^Initial mesh fitting took (\d+) seconds$")),
                ("intensity_prepare_and_fit",re.compile(r"^Mesh fitting took (\d+) seconds$"))]


def extract_fnit(api, record):
    base={key:record.get(key) for key in ("mode","repeat","output")}
    rows=[]
    def add(structure,step,seconds,scope,field):
        if seconds is not None and (not math.isfinite(seconds) or seconds<0):
            raise ValueError("Nonfinite/negative actual timing: "+field)
        rows.append({**base,"method":"fnit","structure":structure,"step":step,"seconds":seconds,
                     "scope":scope,"field":field,"intermediate_cross_dice":None})
    shared=api["initialization"]["shared_preprocessing"]
    add("shared","shared_preprocessing",shared["seconds"],"Input read, shared coarse/parcellation/wmparc, and raw intensity preprocessing if selected; includes actual context observer callback overhead.","initialization/shared_preprocessing/seconds")
    bias=shared.get("intensity_preprocessing",{}).get("bias_correction_seconds")
    if bias is not None:
        add("shared","raw_bias_correction_subset",bias,"Subset of shared preprocessing; synchronized TorchFAST call, do not add again to total.","initialization/shared_preprocessing/intensity_preprocessing/bias_correction_seconds")
    for structure in api["structures"]:
        recipe = api["initialization"][structure]
        timing = recipe.get("timing_seconds", {})
        if timing:
            for original,step in (("alignment","atlas_alignment"),("segmentation_fit","synthetic_fit_and_preparation"),
                                  ("image_preparation","image_preparation"),("intensity_mesh_fit","intensity_fit"),
                                  ("postprocess","postprocess"),("total","recipe_total")):
                add(structure,step,timing.get(original),"Actual independent BS recipe timer; recipe total contains the steps and must not be added again.",f"initialization/{structure}/timing_seconds/{original}")
        else:
            synthetic = recipe.get("segmentation_fit", {})
            working = recipe.get("working_image", {})
            intensity = recipe.get("mesh_solver", {})
            add(structure,"atlas_alignment",None,"TH/HIP have no independently recorded affine timer; missing, not zero.",None)
            add(structure,"synthetic_fit_and_preparation",synthetic.get("seconds"),"Actual fit_segmentation_mesh timer including class/synthetic setup, multiscale _fit and displacement statistics.",f"initialization/{structure}/segmentation_fit/seconds")
            add(structure,"image_preparation",working.get("working_image_preparation_seconds"),"Actual prepare_working_image timer before intensity solver.",f"initialization/{structure}/working_image/working_image_preparation_seconds")
            add(structure,"intensity_multiscale_prepare_fit",intensity.get("total_seconds"),"Sum of synchronized multiscale intensity stage timers; includes per-stage preparation, GEMS fit and post-fit bookkeeping.",f"initialization/{structure}/mesh_solver/total_seconds")
            add(structure,"postprocess",None,"TH/HIP have no independently recorded postprocess timer; missing, not zero.",None)
            add(structure,"recipe_total",recipe.get("seconds"),"Actual recipe.run whole timer; includes atlas loading, affine, synthetic, intensity and postprocess.",f"initialization/{structure}/seconds")
            parts=[synthetic.get("seconds"),working.get("working_image_preparation_seconds"),intensity.get("total_seconds")]
            if recipe.get("seconds") is not None and all(value is not None for value in parts):
                residual=recipe["seconds"]-sum(parts)
                if residual<-.01:raise ValueError("Recipe independent timer totals overlap or exceed recipe time")
                add(structure,"recipe_unallocated_overhead",max(0.,residual),"Derived residual of recorded recipe total minus synthetic timer, working-image preparation and multiscale intensity total; contains atlas load/affine/postprocess and other untimed glue. No allocation to individual missing steps.",f"initialization/{structure}/seconds minus independent recorded subtotals")
        for phase,solver,path in (("synthetic",recipe.get("segmentation_fit",{}).get("mesh_solver",{}),f"initialization/{structure}/segmentation_fit/mesh_solver"),
                                  ("intensity",recipe.get("mesh_solver",{}),f"initialization/{structure}/mesh_solver")):
            for original,step in (("preparation_seconds","multiscale_prepare_subset"),("gems_fit_seconds","multiscale_gems_fit_subset"),("post_fit_seconds","multiscale_postfit_subset"),("total_seconds","multiscale_total_subset")):
                if original in solver:
                    add(structure,phase+"_"+step,solver[original],"Recorded synchronized sum of solver-stage timers; subset of this phase, do not add again to enclosing phase or recipe total.",path+"/"+original)
            for index,stage in enumerate(solver.get("stages", []),1):
                for original in ("preparation_seconds","gems_fit_seconds","post_fit_seconds","total_seconds"):
                    if original in stage:
                        add(structure,f"{phase}_stage{index}_{original}",stage[original],"Actual synchronized solver-stage timer; subdivision of phase subtotals, not an additional independent duration.",path+f"/stages/{index-1}/"+original)
    for step,key,scope in (("compute","compute_seconds","Public pipeline compute including native output merge, before save."),
                           ("save","save_seconds","One shared output-saving timer; not per-structure allocation.")):
        add("all",step,api["timings"].get(key),scope,"timings/"+key)
    for step,key,scope in (("api_total","api_total_seconds","Production public API compute and save; retains actual in-hook observer overhead."),
                           ("process_wall","process_wall_seconds","Popen-to-independent-wait elapsed including imports, metadata hash observer, API, scoring and wrapper finalization; excludes parent GPU/preflight."),
                           ("context_observer_subset","context_observer_seconds","Only callback array hashing and callback JSON write subtotal; source hashes/final wrapper save are additional process overhead; no subtraction to claim observer-free time."),
                           ("source_input_preflight","preflight_identity_seconds","Parent-side immutable source/input/assets identity checking before Popen."),
                           ("gpu_budget_wait","gpu_budget_wait_seconds","Parent-side GPU free-memory checks and wait before Popen.")):
        add("all",step,record.get(key),scope,"run_audit/"+key)
    return rows


def extract_official(explicit, metadata):
    rows=[]
    for run in explicit["runs"]:
        repeat=int(run["run"].removeprefix("official_r"))
        batches=2 if run["structure"]=="hippo-amygdala" else 1
        if len(run["timers"])!=4*batches:
            raise ValueError("Unexpected official explicit timer count")
        for index,timer in enumerate(run["timers"]):
            step,pattern=TIMER_PATTERNS[index%4]
            match=pattern.fullmatch(timer["text"])
            if not match:
                raise ValueError("Official timer text/phase does not match explicit source timer order")
            structure=run["structure"]+("-left" if index//4==0 else "-right") if batches==2 else run["structure"]
            rows.append({"method":"official","mode":"stage","repeat":repeat,"structure":structure,
                "step":step,"seconds":int(match[1]),"scope":"Explicit integer seconds from official process.py timer. Preprocessing includes official initialization; synthetic/intensity include preparation and fit, scope differs from FNIT recipe timers.",
                "field":f"{run['log_path']}:{timer['line']}","output":None,"intermediate_cross_dice":None})
        for structure in (["hippo-amygdala-left","hippo-amygdala-right"] if batches==2 else [run["structure"]]):
            rows.append({"method":"official","mode":"stage","repeat":repeat,"structure":structure,
                "step":"postprocess","seconds":None,"scope":"extract/postprocess/cleanup have no independent official timer in saved run.log; missing, not zero.","field":None,"output":None,"intermediate_cross_dice":None})
    for queue in metadata["runs"]:
        total=0.
        for run in queue["runs"]:
            if run["exit_code"]!=0 or not math.isfinite(run["wall_seconds"]) or run["wall_seconds"]<=0:
                raise ValueError("Official completed queue has invalid wall timing")
            total+=run["wall_seconds"]
            rows.append({"method":"official","mode":"stage","repeat":queue["repeat"],"structure":run["structure"],
                "step":"subregion_process_wall","seconds":run["wall_seconds"],"scope":"Official independent command process wall from completed fresh queue; bilateral hippo-amygdala command contains both hemispheres.","field":"queue/runs/wall_seconds","output":None,"intermediate_cross_dice":None})
        rows.append({"method":"official","mode":"stage","repeat":queue["repeat"],"structure":"all",
            "step":"summed_subregions_wall","seconds":total,"scope":"Sum of BS+TH+bilateral HIP command times on fixed norm/aseg/wmparc; not raw-T1 recon-all and not elapsed of three concurrent repeat queues.","field":"queue/runs/wall_seconds sum","output":None,"intermediate_cross_dice":None})
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-result",type=Path,required=True)
    parser.add_argument("--official-timers",type=Path,required=True)
    parser.add_argument("--official-metadata",type=Path,required=True)
    parser.add_argument("--reconall-lineage",type=Path)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    result=read_json(args.final_result)
    rows=[]
    for record in result["run_audit"]:
        api_path=Path(record["output"])/"api_report.json"
        if sha256(api_path)!=record["api_report_sha256"]:
            raise ValueError("Final API bytes changed after final analysis")
        rows.extend(extract_fnit(read_json(api_path),record))
    explicit,metadata=read_json(args.official_timers),read_json(args.official_metadata)
    rows.extend(extract_official(explicit,metadata))
    args.output_dir.mkdir(parents=True,exist_ok=True)
    output={"script_sha256":sha256(Path(__file__)),"final_result_sha256":sha256(args.final_result),
            "official_timers_sha256":sha256(args.official_timers),"official_metadata_sha256":sha256(args.official_metadata),
            "scope":"Actual timer metadata only; CPU-only; final production vs three fresh official repeats.",
            "intermediate_accuracy":{"status":"not_quantified_without_comparable_saved_stage_labels",
                "explanation":"Only final native/HR label comparisons are quantified here. An objective value, initialization mask Dice, or progress timestamp is not an intermediate cross-implementation segmentation Dice."},
            "rows":rows}
    if args.reconall_lineage:
        lineage=read_json(args.reconall_lineage)
        if not lineage["same_voxel_values"] or not lineage["same_affine_atol1e-5"] or lineage["exit_code"]!=0:
            raise ValueError("Historical recon-all input lineage is not numerically closed")
        stage=[row["seconds"] for row in rows if row["method"]=="official" and row["step"]=="summed_subregions_wall"]
        wall=lineage["historical_reconall_wall_seconds"]
        output["historical_raw_stage_sum"]={"reconall_lineage_sha256":sha256(args.reconall_lineage),
            "historical_reconall_wall_seconds":wall,"summed_observed_wall_seconds_min":wall+min(stage),
            "summed_observed_wall_seconds_max":wall+max(stage),
            "scope":"Historical 2026-09-23 recon-all time plus current norm-stage subregion command sums; separate measured segments, not one current raw-T1 end-to-end rerun. No historical compressed T1 SHA was recorded; retained converted input matches current public T1 values/affine."}
    write_json(args.output_dir/"final_step_timing.json",output)
    columns=["method","mode","repeat","structure","step","seconds","scope","field","output","intermediate_cross_dice"]
    with (args.output_dir/"final_step_timing.tsv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,delimiter="\t",fieldnames=columns)
        writer.writeheader();writer.writerows(rows)
    print(json.dumps({"state":"completed","rows":len(rows),"result_sha256":sha256(args.output_dir/"final_step_timing.json")}))


if __name__=="__main__":main()
