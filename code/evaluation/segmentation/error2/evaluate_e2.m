% This code achieve E2 loss which is introduced by NICE. I competition.
% The E2 loss aims to compensate the disproportion between the apriori probabilities of ¡±iris¡±and ¡±non-iris¡± pixels in the images.
%
% Before evaluation, we must transform the mask predicted by the proposed
% method to binary mask.
clear;clc;

gt_data = 'casia';
path = 'H:\research\Iris\IrisSegNet\IrisNet\2018-6-28\IrisSegNet_final\mask_psp\test\casia\iris_iter_30000\';

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
    
   [tp,fp,tn,fn]=compute_image(msk,gt);
   err_rates(i)=(fp+fn)/2;
   
end

d=mean(err_rates)*100;
disp(d);

% save e2 loss
fid= fopen([path,'mask_binary_error2.txt'],'w');
fprintf(fid,'error2 of %s :  %f%%',gt_data,d);
fclose(fid);