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
% Complex Valued Neural Nework for classification
% Supervised learning with Input data and teaching data
% Data Type: complex numbers
%
% % *?=?hermite conjugate is not implimented!!
%
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

function [wHI, wOH, zO_set] = cvnn (zI_set, zO_teach_set)
% INITIALIZE VALUES
sizeI   = 16 +1;            % number of input neurons
sizeO   = 16;               % number of output neurons
sizeH   = 25 +1;            % number of hidden neurons
k       = 0.2;              % learning constant (paper k = 01)
[s, ~]  = size(zI_set);     % number of signals (row)
zO_set  = zeros(s, sizeO);  % matrix to save the output signals
zHt = zeros;

% INITIALIZEW WEIGHT MATRIX with random values
wHI = rand(sizeH, sizeI);   % weight matrix from hidden to input
wOH = rand(sizeO, sizeH);   % weight matrix from output to hidden
value = zeros(sizeH, sizeI);

deltaEwHI1 = zeros(sizeH, sizeI);
deltaEwHI2 = zeros(sizeH, sizeI);

deltaEwOH1 = zeros(sizeO, sizeH);
deltaEwOH2 = zeros(sizeO, sizeH);

iteration = 3001;

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
        wHzI    = (wHI * zI_set(row, :).').';               % matrix h to i * output of input layer/input zI
        zH      = tanh(abs(wHzI)) .* exp(1i * angle(wHzI)); % activation function

        % calculate the output layer output signal vector zO
        % with the activation function fr(x) = tanh(x)
        wOzH    = (wOH * zH.').';                           % matrix o to i * output of hidden layer zH
        zO      = tanh(abs(wOzH)).* exp(1i * angle(wOzH));  % activation function
        
        
        % calculating the error value
        % compare every element of zI with every element of the respective
        % teacher output
        temp    = abs((zO - zO_teach_set(row, :))).^2;
        er      = (1/2) .* sum( temp ) ;
        
        % update the weights wHI and wOH
        % deltaEwOH 26x16
        % deltaEwIH 17x26
        
        % first part of w
                     % (1 - |zO|^2)   ( |zO| - |z^O| 
                     % cos(arg zO - arg z^O)) |zH|
                     % cos(arg zO - arg z^O - arg wOH)
                     % - |zO| |z^O| sin(arg zO - arg z^O)
                     % |zH| / tanh^1 |zO|
                     % sin(arg zO - arg z^O - arg wOH)
        zOt = zO_teach_set(row, :);
        zI  = zI_set(row, :);
        
        for ii = 1:sizeO
          for jj = 1:sizeH           
            deltaEwOH1(ii,jj) =  (1- abs(zO(ii)).^2).' .* ( (abs(zO(ii)) - abs(zOt(ii))) .* ...
                     cos(angle(zO(ii)) - angle(zOt(ii))) ).' * abs(zH(jj)) .* ...
                     cos(angle(zO(ii)) - angle(zOt(ii)) - angle(wOH(ii,jj))).' - ...
                     abs(zO(ii)) * abs(zOt(ii)) * sin(angle(zO(ii)) - angle(zOt(ii))) .* ...
                     (abs(zH(jj)) / atanh(abs(zO(ii)))).* ...
                     sin(angle(zO(ii)) - angle(zOt(ii) - angle(wOH(ii,jj))));
          end
        end
        
        % a little bit different for the second part of w
        for ii = 1:sizeO
          for jj = 1:sizeH           
            deltaEwOH2(ii,jj) =  (1- abs(zO(ii)).^2).' .* ( (abs(zO(ii)) - abs(zOt(ii))) .* ...
                     cos(angle(zO(ii)) - angle(zOt(ii))) ).' * abs(zH(jj)) .* ...
                     sin(angle(zO(ii)) - angle(zOt(ii)) - angle(wOH(ii,jj))).' + ...
                     abs(zO(ii)) * abs(zOt(ii)) * sin(angle(zO(ii)) - angle(zOt(ii))) .* ...
                     (abs(zH(jj)) / atanh(abs(zO(ii)))).* ...
                     cos(angle(zO(ii)) - angle(zOt(ii) - angle(wOH(ii,jj))));

          end
        end
        
        
        
        % the same as above, but the indexes o and h are
        % replaced with h and i
        
        % zH_teach = (f (zO_teach)*wO_teach)* 
        % *?=?hermite conjugate is not implimented
        for h = 1:sizeH
            zHt(h) = (tanh(zOt * wOH(:,h)).' * exp(1i * angle(zOt * wOH(:,h)))).';
        end
        
        
        for ii = 1:sizeH
          for jj = 1:sizeI           
            deltaEwHI1(ii,jj) =  (1- abs(zH(ii)).^2).' .* ( (abs(zH(ii)) - abs(zHt(ii))) .* ...
                     cos(angle(zH(ii)) - angle(zHt(ii))) ).' * abs(zI(jj)) .* ...
                     cos(angle(zH(ii)) - angle(zHt(ii)) - angle(wHI(ii,jj))).' - ...
                     abs(zH(ii)) * abs(zHt(ii)) * sin(angle(zH(ii)) - angle(zHt(ii))) .* ...
                     (abs(zI(jj)) / atanh(abs(zH(ii)))).* ...
                     sin(angle(zH(ii)) - angle(zHt(ii) - angle(wHI(ii,jj))));
          end
        end
        
        % a little bit different for the second part of w
        for ii = 1:sizeH
          for jj = 1:sizeI           
            deltaEwHI2(ii,jj) =  (1- abs(zH(ii)).^2).' .* ( (abs(zH(ii)) - abs(zHt(ii))) .* ...
                     cos(angle(zH(ii)) - angle(zHt(ii))) ).' * abs(zI(jj)) .* ...
                     sin(angle(zH(ii)) - angle(zHt(ii)) - angle(wHI(ii,jj))).' + ...
                     abs(zH(ii)) * abs(zHt(ii)) * sin(angle(zH(ii)) - angle(zHt(ii))) .* ...
                     (abs(zI(jj)) / atanh(abs(zH(ii)))).* ...
                     cos(angle(zH(ii)) - angle(zHt(ii) - angle(wHI(ii,jj))));
          end
        end

        wHI1 = abs(wHI) - k * deltaEwHI1;
        wOH1 = abs(wOH) - k * deltaEwOH1;
        
        wHI2 = angle(wHI) - k * deltaEwHI2;
        wOH2 = angle(wOH) - k * deltaEwOH2;
        
        wHI = wHI1 .* exp(1).^(1i.* wHI2);
        wOH = wOH1 .* exp(1).^(1i.* wOH2);
        
        % storing zO into zO_set
        zO_set(row, :) = zO;  
    end
        er_matrix(counter) = er;
        counter = counter +1;
        disp(er) 
end

% Printing the Error-Value \ Iteration graph
y = (1:counter-1);
figure
plot(y, er_matrix)
title('ER Value Development')
xlabel('Iteration')
ylabel('ER Value')
axis([0 counter 0 inf])

end