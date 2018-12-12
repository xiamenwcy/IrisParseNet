% This code achieve E1 loss which is introduced by NICE. I competition.
% The E1 loss aims to evaluate the inconsistent pixels between ground truth
% and predicted mask image.
%
% Before evaluation, we must transform the mask predicted by the proposed
% method to binary mask.

clear;clc;
gt_data = 'nice';

path = 'H:\research\Iris\IrisSegNet\IrisNet\2018-6-28\IrisSegNet_final\mask_psp\test\nice\iris_iter_30000\';

msk_dir = [path,'mask_predict_binary\'];

switch(gt_data)
    case 'casia'
        gt_dir = '../mask_gt/CASIA_mask_ground_binary/';
    case 'nice'
        gt_dir = '../mask_gt/NICE_mask_ground_binary/';
    case 'miche'    
        gt_dir = '../mask_gt/MICHE_mask_ground_binary/';
    otherwise
        disp('error')
end

files = dir([msk_dir, '*.bmp']);
n = length(files);
err_rates = zeros(n, 1);

for i = 1:n
    msk = imread([msk_dir, files(i).name]);
    gt = imread([gt_dir, files(i).name]);
    gt = gt(:, :, 1);
    
    err = nnz(xor(msk, gt))/numel(msk);
    err_rates(i) = min(err, 1-err);
end

d=mean(err_rates)*100;
disp(d);

%save result

fid= fopen([path,'mask_binary_error1.txt'],'w');
fprintf(fid,'error1 of %s:  %f%%',gt_data,d);
fclose(fid);