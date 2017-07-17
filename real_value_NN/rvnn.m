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
% Real Valued Neural Nework for classification
% Supervised learning with Input data and teaching data
% Data Type: real numbers (two representing one complex number)
%
% WARNING
% % *?=?hermite conjugate is not implimented!!
%
% INPUT, OUTPUT
% Input: Matrix zI_set
%     row:    Signal Vectors zI
%     column: Signal Values  zI_i
%
% Output: Matrix zO_set
%     row:    Signal Vectors zO
%     column: Signal Values  zO_i
%
% Teacher Signal: Matrix zO_teach_set
%     row:    Signal Vectors zO_teach
%     column: Signal Values  zO_i


function [wHI, wOH, zO_set] = rvnn (zI_set, zO_teach_set)
% initialize values
sizeI   = 32 +2;            % number of input neurons
sizeO   = 32;               % number of output neurons
sizeH   = 50 +2;            % number of hidden neurons
k       = 0.2;              % learning constant (paper k = 01)
[s, ~]  = size(zI_set);     % number of signals (row)
zO_set  = zeros(s, sizeO);  % matrix to save the output signals

% initialize the weight matrix with random values
wHI = rand(sizeH, sizeI);   % weight matrix from hidden to input
wOH = rand(sizeO, sizeH);   % weight matrix from output to hidden

iteration = 30000;
% start er value should be bigger than the beginning of the iteration
counter = 1;
er_matrix = zeros();

while counter < iteration
    er = 0;
    for row = 1:s
        
        % normalizing Input
        if sum(zI_set(row, :)) > 1
            zI_set(row, :) = zI_set(row, :)/ 1000;
        end
        
        % normalizing Output
        if sum(zO_teach_set(row, :)) > 1
            zO_teach_set(row, :) = zO_teach_set(row, :) / 1000;
        end
       
        
        
        % calculate the hidden layer output signal vector zH
        % with the activation function fr(x) = tanh(x)
        wHzI    = (wHI * zI_set(row, :).').';   % matrix h to i * output of input layer/input zI
        zH      = tanh(wHzI);                   % activation function

        % calculate the output layer output signal vector zO
        % with the activation function fr(x) = tanh(x)
        wOzH    = (wOH * zH.').';   % matrix o to i * output of hidden layer zH
        zO      = tanh(wOzH);       % activation function

        % calculating the error value
        % compare every element of zI with every element of the respective
        % teacher output
        temp    = abs((zO - zO_teach_set(row, :))).^2;
        er      = er + (1/2) .* sum( temp ) ;

        % update the weights wHI and wOH
        deltaEwOH = (zO - zO_teach_set(row, :)).' .* (1 - zO.^2).' * zH;

        temp2 = (zO - zO_teach_set(row, :)) .* (1 - zO.^2) * wOH;
        deltaEwHI = ( sum(temp2) ) * (1 - zH.^2).' * zI_set(row, :);

        wHI = wHI - k * deltaEwHI;
        wOH = wOH - k * deltaEwOH;
        
        % storing zO into zO_set
        zO_set(row, :) = zO;  
    end
        er_matrix(counter) = er;
        counter = counter +1;
        disp(er) 
end
y = (1:counter-1);

figure
plot(y, er_matrix)
title('ER Value Development')
xlabel('Iteration')
ylabel('ER Value')
axis([0 160 0 inf])

end