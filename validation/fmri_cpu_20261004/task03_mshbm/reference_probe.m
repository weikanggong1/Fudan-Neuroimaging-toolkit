function reference_probe(binding_path, output_dir)
% Validate original reader ABI and complete real inputs before official timing.
if exist(output_dir, 'dir')
    error('FNITBenchmark:ExistingProbe', 'Preserve every previous attempt');
end
mkdir(output_dir);
cfg = jsondecode(fileread(binding_path));
setenv('CBIG_CODE_DIR', cfg.cbig_dependency_dir);
setenv('CBIG_SD_DIR', cfg.cbig_sd_dir);
startup_path = strrep(cfg.matlab_startup, '''', '''''');
evalin('base', ['run(''' startup_path ''');']);
setenv('CBIG_CODE_DIR', cfg.cbig_clean_dir);
project = fullfile(cfg.cbig_clean_dir, 'stable_projects', 'brain_parcellation', 'Kong2019_MSHBM');
addpath(genpath(fullfile(cfg.cbig_clean_dir, 'utilities', 'matlab')), '-begin');
addpath(genpath(project), '-begin');
vendor = fullfile(cfg.cbig_clean_dir, 'external_packages', 'matlab', 'default_packages', 'cifti-matlab');
addpath(vendor, '-begin');
maxNumCompThreads(8);
required = {'CBIG_MSHBM_parcellation_single_subject', 'CBIG_MSHBM_read_fmri', ...
    'CBIG_ComputeCorrelationProfile', 'CBIG_MSHBM_generate_individual_parcellation', ...
    'CBIG_corr', 'ft_read_cifti', 'ft_write_cifti', 'gifti', 'xmltree'};
resolved = struct();
for index = 1:numel(required)
    location = which(required{index});
    if ~startsWith(location, [cfg.cbig_clean_dir filesep])
        error('FNITBenchmark:ShadowedProbe', 'Reference function resolves outside clean closure');
    end
    resolved.(required{index}) = location;
end
descriptor = fopen(fullfile(output_dir, 'resolved_functions.private.json'), 'w');
fprintf(descriptor, '%s\n', jsonencode(resolved));
fclose(descriptor);
timeseries = cfg.timeseries;
if ischar(timeseries)
    input = timeseries;
else
    input = timeseries{1};
end
source = ft_read_cifti(input);
if ~isfield(source, 'dtseries') || ~isfield(source, 'brainstructure')
    error('FNITBenchmark:ReaderABI', 'Fixed original reader must provide its dtseries/brainstructure fields');
end
vertices = source.brainstructure == 1 | source.brainstructure == 2;
series = single(source.dtseries(vertices, :));
if ~isequal(size(series), [64984, 490])
    error('FNITBenchmark:IncompleteProbe', 'Every fsLR32k vertex and all 490 frames are required');
end
descriptor = fopen(fullfile(output_dir, 'full_cortex.private.float32'), 'w');
fwrite(descriptor, series, 'single');
fclose(descriptor);
prior = load(fullfile(project, 'lib', 'group_priors', 'HCP_40', 'Params_Final.mat'));
if isfield(prior, 'Params')
    prior = prior.Params;
end
if ~isequal(size(prior.mu), [1483,17]) || ~isequal(size(prior.theta), [64984,17])
    error('FNITBenchmark:PriorShape', 'Original HCP40 prior has incorrect shape');
end
report = struct('scope', 'actual original reader and assets; not benchmark timing', ...
                'complete_frames', 490, 'full_vertices', 64984, ...
                'reader_field', 'dtseries', 'matlab_version', version, ...
                'mu_shape', size(prior.mu), 'theta_shape', size(prior.theta), ...
                'required_functions_resolve_to_clean_export', true);
descriptor = fopen(fullfile(output_dir, 'report.public.json'), 'w');
fprintf(descriptor, '%s\n', jsonencode(report));
fclose(descriptor);
fprintf('REFERENCE_READER_PROBE_COMPLETE\n');
end
