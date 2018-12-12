clear;clc;
%before evluation, we need to transform the mask predicted by the proposed method to binary image. 

gt_data = 'nice';
path = 'H:\research\Iris\IrisLocation\2018-10-10\nice\';
iris_dir = [path,'iris_circle_mask_binary\'];
pupil_dir = [path,'pupil_circle_mask_binary\'];

switch(gt_data)
    case 'casia'
        iris_gt_dir = '../edge_gt/CASIA/test/iris_ellipse_mask_binary/';
        pupil_gt_dir = '../edge_gt/CASIA/test/pupil_ellipse_mask_binary/';
    case 'nice'
        iris_gt_dir = '../edge_gt/NICE/test/iris_ellipse_mask_binary/';
        pupil_gt_dir = '../edge_gt/NICE/test/pupil_ellipse_mask_binary/';
    case 'miche'    
        iris_gt_dir = '../edge_gt/MICHE/test/iris_ellipse_mask_binary/';
        pupil_gt_dir = '../edge_gt/MICHE/test/pupil_ellipse_mask_binary/';
    otherwise
        disp('error')
end
iris_files = dir([iris_dir, '*.bmp']);
iris_n = length(iris_files);
iris_Hausdorff = zeros(iris_n, 1);

for i = 1:iris_n
    msk = imread([iris_dir, iris_files(i).name]);
    gt = imread([iris_gt_dir, iris_files(i).name]);
    gt = gt(:, :, 1);
    
    iris_Hausdorff(i) = Hausdorff(msk,gt);
    if iris_Hausdorff(i)==Inf
        disp(iris_files(i).name);
    end
  progressbar( i/iris_n);
end

selected=iris_Hausdorff(iris_Hausdorff ~= Inf) ;
iris_d=mean(selected);

pupil_files = dir([pupil_dir, '*.bmp']);
pupil_n = length(pupil_files);
pupil_Hausdorff = zeros(pupil_n, 1);

for i = 1:pupil_n
    msk = imread([pupil_dir, pupil_files(i).name]);
    gt = imread([pupil_gt_dir, pupil_files(i).name]);
    gt = gt(:, :, 1);
    
    pupil_Hausdorff(i) = Hausdorff(msk,gt);
    if pupil_Hausdorff(i)==Inf
        disp(pupil_files(i).name);
    end
 progressbar( i/pupil_n);
end

selected2=pupil_Hausdorff(pupil_Hausdorff ~= Inf) ;
pupil_d=mean(selected2);
avg_d=(iris_d+pupil_d)/2;

%save result
fid= fopen([path,'circle_hausdorff.txt'],'w');
fprintf(fid,'iris_mdis of %s :  %f\n',gt_data,iris_d);
fprintf(fid,'pupil_mdis of %s :  %f\n',gt_data,pupil_d);
fprintf(fid,'avg_mdis of %s :  %f\n',gt_data,avg_d);
fclose(fid);

