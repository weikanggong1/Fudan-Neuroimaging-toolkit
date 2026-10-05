function run_reference(binding_path, output_dir, num_threads, mode)
% Run the pristine CBIG single-subject reference in an isolated MATLAB process.
% Paths remain in a private JSON binding. mode is 'full' or 'profiles'.
% The coordinator fixes affinity and records the complete MATLAB process time.
if exist(output_dir, 'dir')
    error('FNITBenchmark:ExistingOutput', 'Use a new reference output directory');
end
mkdir(output_dir);
cfg = jsondecode(fileread(binding_path));
maxNumCompThreads(num_threads);
if isfield(cfg, 'cbig_dependency_dir')
    setenv('CBIG_CODE_DIR', cfg.cbig_dependency_dir);
end
if isfield(cfg, 'cbig_sd_dir')
    setenv('CBIG_SD_DIR', cfg.cbig_sd_dir);
end
if isfield(cfg, 'matlab_startup') && ~isempty(cfg.matlab_startup)
    % Upstream startup contains clear and must run in its intended workspace.
    % Keep benchmark bindings and timing state in this function's workspace.
    startup_path = strrep(cfg.matlab_startup, '''', '''''');
    evalin('base', ['run(''' startup_path ''');']);
end
if isfield(cfg, 'dependency_paths')
    paths = as_cell(cfg.dependency_paths);
    for index = 1:numel(paths)
        addpath(paths{index});
    end
end
setenv('CBIG_CODE_DIR', cfg.cbig_clean_dir);
project = fullfile(cfg.cbig_clean_dir, 'stable_projects', ...
                  'brain_parcellation', 'Kong2019_MSHBM');
addpath(genpath(fullfile(cfg.cbig_clean_dir, 'utilities', 'matlab')), '-begin');
addpath(genpath(project), '-begin');
addpath(fullfile(cfg.cbig_clean_dir, 'external_packages', 'matlab', ...
                'default_packages', 'cifti-matlab'), '-begin');
required = {'CBIG_MSHBM_generate_individual_parcellation', ...
            'CBIG_MSHBM_read_fmri', 'CBIG_MSHBM_parcellation_single_subject', ...
            'CBIG_ComputeCorrelationProfile', 'CBIG_corr', 'ft_read_cifti', ...
            'ft_write_cifti', 'gifti', 'xmltree'};
resolved = struct();
for index = 1:numel(required)
    location = which(required{index});
    if ~startsWith(location, [cfg.cbig_clean_dir filesep])
        error('FNITBenchmark:ReferenceShadowed', ...
              'A CBIG reference function resolved outside the clean export');
    end
    resolved.(required{index}) = location;
end
project_dir = fullfile(output_dir, 'project');
if strcmp(mode, 'full')
    params = struct();
    params.project_dir = project_dir;
    params.lh_fMRI_list = as_cell(cfg.timeseries);
    if isfield(cfg, 'censor') && ~isempty(cfg.censor)
        params.censor_list = as_cell(cfg.censor);
    else
        params.censor_list = 'NONE';  % Upstream no-censor sentinel is scalar char.
    end
    params.target_mesh = 'fs_LR_32k';
    params.group_prior = fullfile(project, 'lib', 'group_priors', 'HCP_40', 'Params_Final.mat');
    params.w = num2str(cfg.w);
    params.c = num2str(cfg.c);
    params.overwrite_flag = 0;
    started = tic;
    [lh_labels, rh_labels] = CBIG_MSHBM_parcellation_single_subject(params);
    function_seconds = toc(started);
elseif strcmp(mode, 'profiles')
    profiles = as_cell(cfg.profiles);
    list_dir = fullfile(project_dir, 'profile_list', 'test_set');
    mkdir(list_dir);
    mkdir(fullfile(project_dir, 'priors'));
    copyfile(fullfile(project, 'lib', 'group_priors', 'HCP_40', 'Params_Final.mat'), ...
             fullfile(project_dir, 'priors', 'Params_Final.mat'));
    for index = 1:numel(profiles)
        descriptor = fopen(fullfile(list_dir, sprintf('sess%d.txt', index)), 'w');
        if descriptor < 0
            error('FNITBenchmark:ProfileList', 'Cannot write the profile list');
        end
        fprintf(descriptor, '%s\n', profiles{index});
        fclose(descriptor);
    end
    started = tic;
    [lh_labels, rh_labels] = CBIG_MSHBM_generate_individual_parcellation( ...
        project_dir, 'fs_LR_32k', num2str(numel(profiles)), '17', '1', ...
        num2str(cfg.w), num2str(cfg.c), 'test_set');
    function_seconds = toc(started);
else
    error('FNITBenchmark:Mode', 'Supported modes are full and profiles');
end
if numel(lh_labels) ~= 32492 || numel(rh_labels) ~= 32492
    error('FNITBenchmark:IncompleteLabels', 'Reference must retain every fsLR32k vertex');
end
save(fullfile(output_dir, 'labels.mat'), 'lh_labels', 'rh_labels', '-v7');
report = struct('mode', mode, 'function_seconds', function_seconds, ...
                'matlab_version', version, 'threads', maxNumCompThreads(), ...
                'vertex_count', 64984, 'w', cfg.w, 'c', cfg.c);
descriptor = fopen(fullfile(output_dir, 'report.public.json'), 'w');
fprintf(descriptor, '%s\n', jsonencode(report));
fclose(descriptor);
descriptor = fopen(fullfile(output_dir, 'resolved_functions.private.json'), 'w');
fprintf(descriptor, '%s\n', jsonencode(resolved));
fclose(descriptor);
fprintf('REFERENCE_COMPLETE vertices=64984 seconds=%.6f\n', function_seconds);
end

function result = as_cell(value)
if ischar(value)
    result = {value};
elseif isstring(value)
    result = cellstr(value);
else
    result = value;
end
end
