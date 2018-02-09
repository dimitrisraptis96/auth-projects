%
% SCRIPT: DEMO_MEANSHIFT
%
%   Sample script on usage of mean-shift function.
%
% DEPENDENCIES
%
%   meanshift
%
%


%% CLEAN-UP

clear;
close all;


%% PARAMETERS

% dataset options
basepath = '.';
filename = 'r15';
varX     = 'X';
varL     = 'L';

% mean shift options
h = 10;
optMeanShift.epsilon = 1e-4*h;
optMeanShift.verbose = true;
optMeanShift.display = false;


%% (BEGIN)

fprintf('\n *** begin %s ***\n\n',mfilename);


%% READ DATA

fprintf('...reading csv file...\n')

x = csvread ('../dataset/csv/1024_32.csv');

figure('name', 'original_data')
scatter(x(:,1),x(:,2));

%% PERFORM MEAN SHIFT

fprintf('...computing mean shift...\n')

tic;
y = meanshift( x, h, optMeanShift);
tElapsed = toc;

fprintf('[FINAL]: DONE in %.2f sec\n', tElapsed);


%% SHOW FINAL POSITIONS

figure('name', 'final_local_maxima_points')
scatter(y(:,1),y(:,2));


%% (END)

fprintf('\n *** end %s ***\n\n',mfilename);


%%------------------------------------------------------------
%
% AUTHORS
%
%   Dimitris Floros                         fcdimitr@auth.gr
%
% VERSION
%
%   0.1 - December 29, 2017
%
% CHANGELOG
%
%   0.1 (Dec 29, 2017) - Dimitris
%       * initial implementation
%
% ------------------------------------------------------------