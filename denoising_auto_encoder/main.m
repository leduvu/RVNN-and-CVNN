% Author:       Le Duyen Sandra Vu
% University:   University of Tokyo
%               University of Potsdam
%
% Supervisor:   Akira Hirose (Japan)
%               Manfred Stede (Germany)
% Date:         9/29/2016
% Project:      Neural Networks
% E-Mail:       leduvu@uni-potsdam.de
%
% DESCRIPTION
% main autoencoder program
% gets some sample data and put it into the autoencoder

% get the data, which is converted to complex numbers
data_comp = get_data('mine10_3cm_1');
disp(data_comp)
% start the autoencoder
[weights, zO] = autoen(data_comp);
