% This code achieves  mean Intersection over Union (mIOU) accuracy.
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
iou = zeros(n, 1);

for i = 1:n
    msk = imread([msk_dir, files(i).name]);
    gt = imread([gt_dir, files(i).name]);
    gt = gt(:, :, 1);
    
    [tp,fp,tn,fn]=compute_image(msk,gt);
    if(tp+fn+fp == 0)
        disp(files(i).name)
          iou(i)=1;
    else
          iou(i)=tp/(tp+fn+fp);
    end
end

d=mean(iou)*100;
disp(d);

%save result

fid= fopen([path,'mask_binary_miou.txt'],'w');
fprintf(fid,'mIou of %s :  %f%%',gt_data,d);
fclose(fid);