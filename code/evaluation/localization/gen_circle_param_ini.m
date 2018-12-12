% Example codes:  it's mainly used to generate circle ini file based on
% circle parameters. You can use it to generate your ini file.
% 
path='H:\research\Iris\experiment\TVM\output\NICE1\';
iris_circle_dir=[path,'circles\'];
save_path=[path,'circle_path\'];
if ~exist(save_path,'dir') 
    mkdir(save_path)
end

iris_files = dir([iris_circle_dir, '*.mat']);
iris_n = length(iris_files);

for i = 1:iris_n
    mat = load([iris_circle_dir, iris_files(i).name]);
    [filename, type] = strtok(iris_files(i).name, '-');
    iris(1).exist=true;
    iris(1).center_x = mat.center(1);
    iris(1).center_y = mat.center(2);
    iris(1).radius = mat.radius;
    
    iris(2).exist=true;
    iris(2).center_x = mat.center_p(1);
    iris(2).center_y = mat.center_p(2);
    iris(2).radius = mat.radius_p;
   
    save_filename = [save_path,filename,'.ini'];
  struct2ini(save_filename,iris);
  progressbar( i/iris_n);
end