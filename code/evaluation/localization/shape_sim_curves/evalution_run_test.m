clear;clc;
gt_data = 'nice';
pupil=0; % set drawing object: pupil edge (iris inner boundary) or iris edge (iris outer boundary)
if(pupil)
    algo_folder1=['H:\research\Iris\experiment\TVM\output\',gt_data,'\pupil_circle_mask_binary\'];  %pupil_circle_mask_binary
    dilation_path=['H:\research\Iris\IrisParseNet\IrisNet\2018-6-28\IrisParseNet_final\vgg_dilation\test\',gt_data,'\iris_iter_30000\'];
    myalgo1=[dilation_path,'pupil_circle_mask_binary\'];
    myalgo2=['H:\research\Iris\IrisParseNet\IrisNet\2018-6-28\IrisParseNet_final\vgg_psp\test\',gt_data,'\iris_iter_30000\pupil_circle_mask_binary\'];
  
    algo_set={algo_folder1,myalgo1,myalgo2};
    
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
    
    img_list=[];
    
    files=dir([pupil_gt_dir,'*.bmp']);
    n = length(files);
    m=length(algo_set);
    hist_dist_v=zeros(16,m);
    
    
    for j=1:n
        [filename, type] = strtok(files(j).name, '.');
        gt = imread([pupil_gt_dir,files(j).name]);
        gt = gt(:, :, 1);
        for i=1:m
            algo_filename=strcat(algo_set{i},filename,'.bmp');
            msk = imread(algo_filename);
            iris_Hausdorff= Hausdorff(msk,gt);
            if( iris_Hausdorff ~= Inf)
                for k=1:16
                    if iris_Hausdorff<=(k-1)
                        hist_dist_v(k,i)=hist_dist_v(k,i)+1;
                    end
                end
            else
                disp(algo_filename);
            end
        end
        progressbar(j/n);
    end
    
    hist_dist_v=hist_dist_v./n;
    
    c = [[0,0.45,0.74];[1,0,1];[0,1,0]];       %directly set drawing color
    %c = rand(m,3);  % or randomly set drawing color
    for i=1:m
        plot(0:1:15,hist_dist_v(:,i),'color',c(i,:),'LineWidth',2);
        hold on;
    end
    hold off;
    xlabel('Hausdorff Distance','FontWeight','bold','FontSize',16,'FontName','Times New Roman');
    ylabel('Detection Rate','FontWeight','bold','FontSize',16,'FontName','Times New Roman');
    legend({'RTV-$L^1$','IrisParseNet(ASPP)','IrisParseNet(PSP)'},'Interpreter','LaTex');
    grid on;
    % axis tight;
    axis([0,15,0,1]);
    hgsave(gcf,[dilation_path,gt_data,'_pupil_total_hausdorff.fig']);
else
    algo_folder1=['H:\research\Iris\experiment\TVM\output\',gt_data,'\iris_circle_mask_binary\'];  %pupil_circle_mask_binary
    dilation_path=['H:\research\Iris\IrisParseNet\IrisNet\2018-6-28\IrisParseNet_final\vgg_dilation\test\',gt_data,'\iris_iter_30000\'];
    myalgo1=[dilation_path,'iris_circle_mask_binary\'];
    myalgo2=['H:\research\Iris\IrisParseNet\IrisNet\2018-6-28\IrisParseNet_final\vgg_psp\test\',gt_data,'\iris_iter_30000\iris_circle_mask_binary\'];
    
    algo_set={algo_folder1,myalgo1,myalgo2};
    
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
    
    img_list=[];
    
    files=dir([iris_gt_dir,'*.bmp']);
    n = length(files);
    m=length(algo_set);
    hist_dist_v=zeros(16,m);
    
    
    for j=1:n
        [filename, type] = strtok(files(j).name, '.');
        gt = imread([iris_gt_dir,files(j).name]);
        gt = gt(:, :, 1);
        for i=1:m
            algo_filename=strcat(algo_set{i},filename,'.bmp');
            msk = imread(algo_filename);
            iris_Hausdorff= Hausdorff(msk,gt);
            if( iris_Hausdorff ~= Inf)
                for k=1:16
                    if iris_Hausdorff<=(k-1)
                        hist_dist_v(k,i)=hist_dist_v(k,i)+1;
                    end
                end
            else
                disp(algo_filename);
            end
        end
        progressbar(j/n);
    end
    
    hist_dist_v=hist_dist_v./n;
    
    c = [[0,0.45,0.74];[1,0,1];[0,1,0]];        %directly set drawing color
    %c = rand(m,3);  % or randomly set drawing color
    for i=1:m
        plot(0:1:15,hist_dist_v(:,i),'color',c(i,:),'LineWidth',2);
        hold on;
    end
    hold off;
    xlabel('Hausdorff Distance','FontWeight','bold','FontSize',16,'FontName','Times New Roman');
    ylabel('Detection Rate','FontWeight','bold','FontSize',16,'FontName','Times New Roman');
    legend({'RTV-$L^1$','IrisParseNet(ASPP)','IrisParseNet(PSP)'},'Interpreter','LaTex');
    grid on;
    % axis tight;
    axis([0,15,0,1]);
    hgsave(gcf,[dilation_path,gt_data,'_iris_total_hausdorff.fig']);
    
end