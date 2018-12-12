%before evluation, we need to transform the mask predicted by the proposed method to binary image. 

clear;clc;
msk_dir = 'H:\research\Iris\IrisLocation\2018-10-10\nice\pupil_circle_mask\';  %iris_circle_mask  pupil_circle_mask  
save_dir='H:\research\Iris\IrisLocation\2018-10-10\nice\pupil_circle_mask_binary\';%iris_circle_mask_binary  pupil_circle_mask_binary
files = dir([msk_dir, '*.png']);
n = length(files);
if ~exist(save_dir,'dir') 
    mkdir(save_dir)
end

for i = 1:n
    [filename, type] = strtok(files(i).name, '.');
    msk = imread([msk_dir, filename,'.png']);    
    msk=im2bw(msk,0.5);
    imwrite(msk,[save_dir, filename,'.bmp']);
    progressbar(i/n);
end
 