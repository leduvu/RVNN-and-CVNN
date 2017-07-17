% CORRUPTING THE INPUT 
% stochastic mapping - some Values will be set to 0  
% x -> xC
% BUT loss should be computed for L(x,xC') and NOT L(xC,xC')
%
% Input format: [1;2;3]

function data_comp = corrupt(data_comp)

    % get the size and set the percentage of the corruption
    [s, ~]      = size(data_comp);
    percentage  = 0.2;
    
    for value = 1:s
        r = rand;
        if r < percentage
            data_comp(value) = 0;
        disp(r)
        end
    end
    
end