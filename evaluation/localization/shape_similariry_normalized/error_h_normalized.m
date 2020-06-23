clear;clc;
%评价之前gt和预测结果必须先转换为二值图像
gt_data = 'nice';

path = 'H:\research\Iris\IrisLocation\2018-10-10\nice\';
save_name = 'circle_hausdorff_edge_n.txt';  

iris_dir = [path,'iris_boundary_circle_edge\'];
pupil_dir = [path,'pupil_boundary_circle_edge\'];

% 指定虹膜内外边界的ground truth路径
switch(gt_data)
    case 'CASIA_distance'
        iris_gt_dir = 'J:\pr_data\segmentation_localization\CASIA-distance\test\iris_edge\';
        pupil_gt_dir = 'J:\pr_data\segmentation_localization\CASIA-distance\test\pupil_edge\';
    case 'miche'
        iris_gt_dir = 'J:\pr_data\segmentation_localization\MICHE\test\iris_edge\';
        pupil_gt_dir = 'J:\pr_data\segmentation_localization\MICHE\test\pupil_edge\';
    case 'nice'
        iris_gt_dir = 'J:\pr_data\segmentation_localization\NICE\NICE1\test\iris_edge\';
        pupil_gt_dir = 'J:\pr_data\segmentation_localization\NICE\NICE1\test\pupil_edge\';
    otherwise
        disp('error')
end
iris_files = dir([iris_gt_dir, '*.png']);
iris_n = length(iris_files);
iris_Hausdorff = zeros(iris_n, 1);

for i = 1:iris_n
    msk = imread([iris_dir, iris_files(i).name]);
    gt = imread([iris_gt_dir, iris_files(i).name]);
    gt = gt(:, :, 1);
    [h,w,c]=size(gt); 
    % 《Learning to Find Eye Region Landmarks for Remote Gaze Estimation in Unconstrained Settings》
    % 受上述启发，归一化的参考为宽度。
    iris_Hausdorff(i) = Hausdorff(msk,gt)/w;
    if iris_Hausdorff(i)==Inf
        disp(iris_files(i).name);
    end
    progressbar( i/iris_n);
end

selected=iris_Hausdorff(iris_Hausdorff ~= Inf) ;
iris_d=mean(selected);

pupil_files = dir([pupil_gt_dir, '*.png']);
pupil_n = length(pupil_files);
pupil_Hausdorff = zeros(pupil_n, 1);

for i = 1:pupil_n
    msk = imread([pupil_dir, pupil_files(i).name]);
    gt = imread([pupil_gt_dir, pupil_files(i).name]);
    gt = gt(:, :, 1);
    [h,w,c]=size(gt); 
        
    pupil_Hausdorff(i) = Hausdorff(msk,gt)/w;
    if pupil_Hausdorff(i)==Inf
        disp(pupil_files(i).name);
    end
    progressbar( i/pupil_n);
end

selected2=pupil_Hausdorff(pupil_Hausdorff ~= Inf) ;
pupil_d=mean(selected2);
avg_d=(iris_d+pupil_d)/2;
%写文件

fid= fopen([path,save_name],'w');
fprintf(fid,'iris_mdis of %s :  %f\n',gt_data,iris_d);
fprintf(fid,'pupil_mdis of %s :  %f\n',gt_data,pupil_d);
fprintf(fid,'avg_mdis of %s :  %f\n',gt_data,avg_d);
fclose(fid);
