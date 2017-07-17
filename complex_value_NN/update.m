% COST should be the mean of all error values calculated from each input signal

% UPDATE via STOCHASTIC GRADIENT DESCENT
% sgd = delta er / delta [w, bI, bH]
% new[w, bI, bH] = old[w, bI, bH] - learning rate * sgd


function 
         zI  = data_comp;  
         
            % first part of w
                             % (1 - |zO|^2)   ( |zO| - |z^O| 
                             % cos(arg zO - arg z^O)) |zH|
                             % cos(arg zO - arg z^O - arg weights)
                             % - |zO| |z^O| sin(arg zO - arg z^O)
                             % |zH| / tanh^1 |zO|
                             % sin(arg zO - arg z^O - arg weights)             
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

end