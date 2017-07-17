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
% gets the data and converts them to complex numbers
%
% INPUT
% Two columns with 202906 lines
%
% freq. : 8 GHz - 12 GHz  -> 100 points
% Data has 2 columns Amplitude (x) and Phase (y) || 21 x 21 matrix  = 441
% f = 1 ... 101
%
% x = 1 y = 1 f = 1        A    P
% x = 2 y = 1 f = 1        A    P
%
% convert to complex number:    dB2mag(x) ang(y) ??

function data_comp = get_data(dataName)

    % specify the path of the file
    dataName = ['.\data\' dataName '.txt'];

    % open the filestream, read the file and then close the filestream
    % size: 202906 x 2
    fid1 = fopen(dataName);
    a = fscanf(fid1,'%g %g',[2 inf]);
    data = a';
    fclose(fid1);

    % complexIN = db2mag(data(:,1)) .* angle(data(:,2));???

    % convert to complex number
    % size: 202906 x 1 
    data_comp = (10.^(data(:,1) / 20 )) .* cos(data(:,2) * pi / 180 ) + 1i * ((10.^(data(:,1) / 20)) .* sin(data(:,2) * pi / 180 ));
end