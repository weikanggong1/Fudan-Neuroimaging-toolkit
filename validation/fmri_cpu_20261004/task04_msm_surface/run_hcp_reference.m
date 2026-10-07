function run_hcp_reference(binding_path, output_dir, num_threads)
% Invoke unmodified HCP v4.7.0 functions with their matching original I/O.
wrapper_started = tic;
cfg = jsondecode(fileread(binding_path));
if ~exist(output_dir, 'dir'), mkdir(output_dir); end
% -singleCompThread disables the setter even when setting one. CPU1 is
% already enforced by that flag plus inherited taskset affinity.
effective_threads = 1;
if num_threads > 1
    maxNumCompThreads(num_threads);
    effective_threads = maxNumCompThreads();
end
set(0, 'DefaultFigureVisible', 'off');
addpath(genpath(fullfile(cfg.reference_root, 'global', 'matlab')));
addpath(fullfile(cfg.reference_root, 'MSMAll', 'scripts'), '-begin');
required = {'ComputeVN', 'MSMregression', 'ciftiopen', 'ciftisave', 'ciftisavereset', ...
            'demean', 'normalise'};
resolved = struct();
for index = 1:numel(required)
    location = which(required{index});
    if ~startsWith(location, cfg.reference_root)
        error('FNITBenchmark:ReferenceShadowed', 'Original HCP function resolved outside fixed export');
    end
    resolved.(required{index}) = location;
end
preparation_seconds = 0;
if strcmp(cfg.operation, 'variance_normalization')
    function_started = tic;
    ComputeVN(cfg.clean_dtseries, 'NONE', cfg.ica_timecourses, cfg.noise_components, ...
              fullfile(output_dir, 'VN.dscalar.nii'), cfg.wb_command);
    function_seconds = toc(function_started);
else
    data_file = cfg.clean_dtseries;
    if ~isempty(cfg.variance_normalization)
        % Literal SingleSubjectConcat.sh lines 354 and 453-456: intermediate
        % MEAN is saved by Workbench, before demeaning and division by VN.
        preparation_started = tic;
        mean_file = fullfile(output_dir, 'mean.dscalar.nii');
        data_file = fullfile(output_dir, 'normalized.dtseries.nii');
        checked_system(sprintf('%s -cifti-reduce %s MEAN %s', ...
                               quote(cfg.wb_command), quote(cfg.clean_dtseries), quote(mean_file)));
        checked_system(sprintf(['%s -cifti-math "((TCS - Mean)) / max(VN,0.001)" %s ' ...
                                '-var TCS %s -var Mean %s -select 1 1 -repeat ' ...
                                '-var VN %s -select 1 1 -repeat'], ...
                               quote(cfg.wb_command), quote(data_file), quote(cfg.clean_dtseries), ...
                               quote(mean_file), quote(cfg.variance_normalization)));
        preparation_seconds = toc(preparation_started);
    end
    params = 'NO';
    if strcmp(cfg.method, 'WRN')
        params = fullfile(output_dir, 'wrn_parameters.private.txt');
        descriptor = fopen(params, 'w');
        fprintf(descriptor, '%s\n', cfg.vertex_area, cfg.left_midthickness, cfg.right_midthickness);
        paths = as_cell(cfg.low_dimensional_maps);
        for index = 1:numel(paths), fprintf(descriptor, '%s\n', paths{index}); end
        fclose(descriptor);
    end
    function_started = tic;
    MSMregression(cfg.reference_maps, data_file, cfg.component_indices_file, ...
                  fullfile(output_dir, 'individual_maps'), ...
                  fullfile(output_dir, 'component_weights.dscalar.nii'), ...
                  cfg.wb_command, cfg.method, params, 'NO', cfg.nTPsForSpectra, 'NO', 'NO');
    function_seconds = toc(function_started);
end
report = struct('validation_complete', true, 'operation', cfg.operation, ...
                'matlab_version', version, 'threads', effective_threads, ...
                'function_seconds', function_seconds, 'preparation_seconds', preparation_seconds, ...
                'wrapper_seconds', toc(wrapper_started));
descriptor = fopen(fullfile(output_dir, 'reference_report.public.json'), 'w');
fprintf(descriptor, '%s\n', jsonencode(report)); fclose(descriptor);
descriptor = fopen(fullfile(output_dir, 'resolved_functions.private.json'), 'w');
fprintf(descriptor, '%s\n', jsonencode(resolved)); fclose(descriptor);
fprintf('HCP_REFERENCE_COMPLETE operation=%s seconds=%.6f\n', cfg.operation, function_seconds);
end

function value = quote(value)
% These authorized fixed paths must not contain shell quotes or line breaks.
if any(value == char(39)) || any(value == char(10)) || any(value == char(13))
    error('FNITBenchmark:UnsafePath', 'Reference path contains unsupported shell characters');
end
value = [char(39), value, char(39)];
end

function checked_system(command)
[status, output] = system(command);
if status ~= 0, error('FNITBenchmark:OriginalCommand', 'Original Workbench command failed: %s', output); end
end

function result = as_cell(value)
if ischar(value), result = {value}; elseif isstring(value), result = cellstr(value); else, result = value; end
end
