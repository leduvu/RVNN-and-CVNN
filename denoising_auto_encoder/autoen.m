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
% Autoencoder for feature extraction
% Un-supervised learning with Input data only
% Data Type: complex numbers
%
% WARNING
% % *?=?hermite conjugate is not implimented!!
% % biases are not implemented!!
%
% INPUT 
% complex numbers of the format [x; y; z; ...]
% Have to be 202906 values (size can be changed via sizeI and sizeO in this file)
%
% Iteration         = 3
% learning constant = 0.2
% hidden layer      = 10

function [weights, zO] = autoen(data_comp)
sizeI   = 202906;                   % number of input neurons
sizeO   = 202906;                   % number of output neurons
sizeH   = 10;                       % number of hidden neurons
weights     = rand(sizeI, sizeH);   % weight matrix from visible to hidden
bI      = zeros(1,sizeI);           % bias for input vector
bH      = zeros(1,sizeH);           % bias for hidden verctor
k       = 0.2;                      % learning constant

% initialize partial derivative for first and second part of the complex number
deltaEweights1 = zeros(sizeI, sizeH);
deltaEweights2 = zeros(sizeI, sizeH);

iteration = 3;
counter = 1;
er_matrix = zeros();

disp('start autoencoder')
while counter < iteration
    for row = 1:sizeI
    
        % normalizing Input
        if sum(data_comp) > 1
            data_comp = data_comp/ 1000;
        end
        % ENCODE
        % y = s(Wx+b)
        % s is a non-linearity like sigmoid
        xI = (weights.'*data_comp).' +bH;
        zH = tanh(abs(xI)).* exp(1i * angle(xI));

        % DECODE
        % z = s(W'y+b')
        xH = (weights*zH.').' +bI;
        zO = tanh(abs(xH)).* exp(1i * angle(xH));


        % RECONSTRUCTION ERROR
        % squarred error OR cross entropy
        temp    = abs(zO - data_comp.').^2;
        er      = (1/2) .* sum( temp ) ;

        % COST should be the mean of all error values calculated from each input signal

        % UPDATE via STOCHASTIC GRADIENT DESCENT
        % sgd = delta er / delta [w, bI, bH]
        % new[w, bI, bH] = old[w, bI, bH] - learning rate * sgd
                % first part of w
                             % (1 - |zO|^2)   ( |zO| - |z^O| 
                             % cos(arg zO - arg z^O)) |zH|
                             % cos(arg zO - arg z^O - arg weights)
                             % - |zO| |z^O| sin(arg zO - arg z^O)
                             % |zH| / tanh^1 |zO|
                             % sin(arg zO - arg z^O - arg weights)

         zI  = data_comp;                
                         
            for ii = 1:sizeO
              for jj = 1:sizeH           
                deltaEweights1(ii,jj) =  (1- abs(zO(ii)).^2).' .* ( (abs(zO(ii)) - abs(zI(ii))) .* ...
                         cos(angle(zO(ii)) - angle(zI(ii))) ).' * abs(zH(jj)) .* ...
                         cos(angle(zO(ii)) - angle(zI(ii)) - angle(weights(ii,jj))).' - ...
                         abs(zO(ii)) * abs(zI(ii)) * sin(angle(zO(ii)) - angle(zI(ii))) .* ...
                         (abs(zH(jj)) / atanh(abs(zO(ii)))).* ...
                         sin(angle(zO(ii)) - angle(zI(ii) - angle(weights(ii,jj))));
              end
            end

            % a little bit different for the second part of w
            for ii = 1:sizeO
              for jj = 1:sizeH           
                deltaEweights2(ii,jj) =  (1- abs(zO(ii)).^2).' .* ( (abs(zO(ii)) - abs(zI(ii))) .* ...
                         cos(angle(zO(ii)) - angle(zI(ii))) ).' * abs(zH(jj)) .* ...
                         sin(angle(zO(ii)) - angle(zI(ii)) - angle(weights(ii,jj))).' + ...
                         abs(zO(ii)) * abs(zI(ii)) * sin(angle(zO(ii)) - angle(zI(ii))) .* ...
                         (abs(zH(jj)) / atanh(abs(zO(ii)))).* ...
                         cos(angle(zO(ii)) - angle(zI(ii) - angle(weights(ii,jj))));

              end
            end

     weights1       = abs(weights) - k * deltaEweights1;
     weights2       = angle(weights) - k * deltaEweights2;
     weights_grad   = weights1 .* exp(1).^(1i.* weights2);

     weights        = weights - k * weights_grad;
     %bI     = bI  - k * bI_grad;
     %bH     = bH  - k * bH_grad;
     
     disp(er)
     er_matrix(counter) = er;
     counter = counter +1;
    end
     
end

y = (1:counter-1);

figure
plot(y, er_matrix)
title('ER Value Development')
xlabel('Iteration')
ylabel('ER Value')
axis([0 counter 0 inf])


% Plot real and imaginary parts
figure;
plot([real(weights), imag(weights)])
end
