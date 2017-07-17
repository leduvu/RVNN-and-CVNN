% TEST VALUES IDENTITY EXPERIMENT WITH RANDOM NUMBERS
% zI1 =  [1+1i 1+2i 1+3i 1+4i 1+1i 1+2i 1+3i 1+4i 1+1i 1+2i 1+3i 1+4i 1+1i 1+2i 1+3i 1+4i 1];
% zI2 =  [1+6i 1+5i 1+7i 1+8i 1+6i 1+5i 1+7i 1+8i 1+6i 1+5i 1+7i 1+8i 1+6i 1+5i 1+7i 1+8i 1];
% zI3 =  [1+6i 1+3i 1+3i 1+9i 1+6i 1+3i 1+3i 1+9i 1+6i 1+3i 1+3i 1+9i 1+6i 1+3i 1+3i 1+9i 1];
% zI_set = [zI1; zI2; zI3];
% zO_teach1 = [1+1i 1+2i 1+3i 1+4i 1+1i 1+2i 1+3i 1+4i 1+1i 1+2i 1+3i 1+4i 1+1i 1+2i 1+3i 1+4i];
% zO_teach2 = [1+6i 1+5i 1+7i 1+8i 1+6i 1+5i 1+7i 1+8i 1+6i 1+5i 1+7i 1+8i 1+6i 1+5i 1+7i 1+8i];
% zO_teach3 = [1+6i 1+3i 1+3i 1+9i 1+6i 1+3i 1+3i 1+9i 1+6i 1+3i 1+3i 1+9i 1+6i 1+3i 1+3i 1+9i];
% zO_teach_set = [zO_teach1; zO_teach2; zO_teach3];


% TEST VALUES RANDOM NUMBERS
zI1 =  [1+1i 1+2i 1+3i 1+4i 1+1i 1+2i 1+3i 1+4i 1+1i 1+2i 1+3i 1+4i 1+1i 1+2i 1+3i 1+4i 1];
zI2 =  [1+6i 1+5i 1+7i 1+8i 1+6i 1+5i 1+7i 1+8i 1+6i 1+5i 1+7i 1+8i 1+6i 1+5i 1+7i 1+8i 1];
zI3 =  [1+6i 1+3i 1+3i 1+9i 1+6i 1+3i 1+3i 1+9i 1+6i 1+3i 1+3i 1+9i 1+6i 1+3i 1+3i 1+9i 1];
zI_set = [zI1; zI2; zI3];
zO_teach1 = [1+1i 1+2i 1+3i 1+4i 1+1i 1+2i 1+3i 1+4i 1+1i 1+2i 1+3i 1+4i 1+1i 1+2i 1+3i 1+4i];
zO_teach2 = [1+6i 1+3i 1+3i 1+9i 1+6i 1+3i 1+3i 1+9i 1+6i 1+3i 1+3i 1+9i 1+6i 1+3i 1+3i 1+9i];
zO_teach3 = [1+6i 1+5i 1+7i 1+8i 1+6i 1+5i 1+7i 1+8i 1+6i 1+5i 1+7i 1+8i 1+6i 1+5i 1+7i 1+8i];
zO_teach_set = [zO_teach1; zO_teach2; zO_teach3];

% start deep learning rvnn (real value neural network)
[wHI, wOH, zO_set] = cvnn(zI_set, zO_teach_set);

% print calculated weights and output signals
%disp(wHI); disp(wOH'); disp(zO_set);


% % % TEST VALUES IDENTITY EXPERIMENT WITH AMPLITUDE TIME 
% % % z = 4x4   s = 16 I = 16+1
% SA  = 4;
% St  = 4;
% I   = 16;
% ss   = 1;
% zI_matrix1 = zeros(16,16);
% add_value = ones(1,16);
% 
% for sA = 1:SA
%     for st = 1:St
%         for ii = 1:I
%             zI_matrix1(ii,ss) =  (sA / SA+1) * exp(1i * ( st / (2*St) + (ii / I) ) * 2 * pi);
%         end
%         ss = ss + 1;
%     end
% end
% 
% zI_matrix = [zI_matrix1;add_value];
%    
% zI_set = [zI_matrix(:,1).'; zI_matrix(:,2).'; zI_matrix(:,3).'; zI_matrix(:,4).'; zI_matrix(:,5).'; zI_matrix(:,6).'; zI_matrix(:,7).'; zI_matrix(:,8).'; zI_matrix(:,9).'; zI_matrix(:,10).'; zI_matrix(:,11).'; zI_matrix(:,12).'; zI_matrix(:,13).'; zI_matrix(:,14).'; zI_matrix(:,15).'; zI_matrix(:,16).'];
% 
% zO_teach_set = [zI_matrix1(:,1).'; zI_matrix1(:,2).'; zI_matrix1(:,3).'; zI_matrix1(:,4).'; zI_matrix1(:,5).'; zI_matrix1(:,6).'; zI_matrix1(:,7).'; zI_matrix1(:,8).'; zI_matrix1(:,9).'; zI_matrix1(:,10).'; zI_matrix1(:,11).'; zI_matrix1(:,12).'; zI_matrix1(:,13).'; zI_matrix1(:,14).'; zI_matrix1(:,15).'; zI_matrix1(:,16).'];
