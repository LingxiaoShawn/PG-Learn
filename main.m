fprintf('PG-LEARN: Adding directories to current path...\n');

% make sure we add the correct folders even if this file is
% not called from the current folder
fileName = mfilename();
filePath = mfilename('fullpath');
filePath = filePath(1:end-size(fileName, 2));

% Add folders to current path
path(genpath([filePath 'util']), path);
path(genpath([filePath 'pglearn']), path);
path(genpath([filePath 'baselines']), path);
path(genpath([filePath 'examples']), path);

fprintf('Done!\n');

% clear variables
clearvars fileName filePath