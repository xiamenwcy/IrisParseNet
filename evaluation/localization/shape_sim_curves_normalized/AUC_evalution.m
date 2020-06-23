%https://blog.csdn.net/u011681952/article/details/81985559
%https://blog.csdn.net/liweibin1994/article/details/79462554
clear;clc;
hausdorff_mat_file='F:\research\Iris\CASIA_hausdorff.mat';
hausdorff_mat=load(hausdorff_mat_file);
hist_dist_v=hausdorff_mat.hist_dist_v;
%[n_interval,n_algorithm]=size(hist_dist_v);
threshold=0.01;
auc = sum(hist_dist_v)*threshold; 
disp(auc);
