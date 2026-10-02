# HIP topology diagnostics

five partial HIP real-data diagnostic cases with declared solver profiles; morphology fixed before reference scoring; no solver change by observation; highres scores on candidate fit grid only

|Case|Rule|HIP weighted native Dice|Amyg weighted native Dice|AAA native voxels|AAA HR voxels|AAA native Dice|
|---|---|---:|---:|---:|---:|---:|
|stage_right_baseline_fast|original_lcc6|0.830807|0.929114|22|621|0.9565217391304348|
|stage_right_baseline_fast|closing_ball1_selection_only|0.830807|0.929114|22|624|0.9565217391304348|
|stage_right_baseline_fast|dilation_ball1_selection_only|0.830532|0.929114|22|632|0.9565217391304348|

stage_right_baseline_fast detached AAA gap: `{"detached_AAA_voxels": 13, "minimum_taxicab_centre_distance_voxels": 2, "minimum_empty_voxels_in_taxicab_route": 1, "minimum_euclidean_centre_distance_mm": 0.4713998067458237}`

|stage_right_joint_fast|original_lcc6|0.853946|0.920830|0|1|0.0|
|stage_right_joint_fast|closing_ball1_selection_only|0.853813|0.937903|21|616|0.9333333333333333|
|stage_right_joint_fast|dilation_ball1_selection_only|0.853660|0.937903|21|622|0.9333333333333333|

stage_right_joint_fast detached AAA gap: `{"detached_AAA_voxels": 626, "minimum_taxicab_centre_distance_voxels": 2, "minimum_empty_voxels_in_taxicab_route": 1, "minimum_euclidean_centre_distance_mm": 0.4713998067458237}`

|stage_right_balanced|original_lcc6|0.902432|0.928916|0|0|0.0|
|stage_right_balanced|closing_ball1_selection_only|0.902432|0.945989|21|624|0.9333333333333333|
|stage_right_balanced|dilation_ball1_selection_only|0.902442|0.945989|21|624|0.9333333333333333|

stage_right_balanced detached AAA gap: `{"detached_AAA_voxels": 631, "minimum_taxicab_centre_distance_voxels": 2, "minimum_empty_voxels_in_taxicab_route": 1, "minimum_euclidean_centre_distance_mm": 0.4713998067458237}`

|stage_left_balanced|original_lcc6|0.781051|0.868861|19|640|0.7619047619047619|
|stage_left_balanced|closing_ball1_selection_only|0.781051|0.868861|19|640|0.7619047619047619|
|stage_left_balanced|dilation_ball1_selection_only|0.781051|0.868861|19|641|0.7619047619047619|

stage_left_balanced detached AAA gap: `{"detached_AAA_voxels": 1, "minimum_taxicab_centre_distance_voxels": 3, "minimum_empty_voxels_in_taxicab_route": 2, "minimum_euclidean_centre_distance_mm": 0.7453485389400049}`

|raw_left_balanced|original_lcc6|0.787963|0.879666|18|630|0.7368421052631579|
|raw_left_balanced|closing_ball1_selection_only|0.787963|0.879666|18|631|0.7368421052631579|
|raw_left_balanced|dilation_ball1_selection_only|0.787784|0.879666|18|636|0.7368421052631579|

raw_left_balanced detached AAA gap: `{"detached_AAA_voxels": 6, "minimum_taxicab_centre_distance_voxels": 2, "minimum_empty_voxels_in_taxicab_route": 1, "minimum_euclidean_centre_distance_mm": 0.47139980674582377}`

