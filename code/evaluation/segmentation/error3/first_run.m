% This code achieve precision,recall,f_measure.
% We need to run first_run.m firstly, then run sceond_run.m. 
% Before evaluation, we must transform the mask predicted by the proposed
% method to binary mask.
clear;clc;

path = 'H:\research\Iris\IrisSegNet\IrisNet\2018-6-28\IrisSegNet_final\mask_psp\test\nice\iris_iter_30000\';
msk_dir = [path,'mask_predict_binary\'];
%gt_dir = '../mask_gt/CASIA_mask_ground_binary/';
%gt_dir = '../mask_gt/MICHE_mask_ground_binary/';
gt_dir = '../mask_gt/NICE_mask_ground_binary/';
save_dir=path;
 
cmdpath='software/bin/maskcmpprf.exe';

% save result
fid= fopen([save_dir,'mask_binary_prf1.txt'],'w');
files = dir([msk_dir, '*.bmp']);
n = length(files);
 

for i = 1:n
    msk_file=[msk_dir, files(i).name];
    gt_file = [gt_dir, files(i).name];
    command=[cmdpath,' -i ',gt_file,' ',msk_file,' ','-q'];
   [status, results]=system(command);
   if(status==0)
      fprintf(fid,'%s  %s',files(i).name,results);
   else
      fprintf('%s',files(i).name);
   end
      progressbar( i/n);
end
fclose(fid);