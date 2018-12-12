
clear;clc;
path = 'H:\research\Iris\IrisSegNet\IrisNet\2018-6-28\IrisSegNet_final\mask_psp\test\nice\iris_iter_30000\';
filepath=[path,'mask_binary_prf1.txt'];
savepath=[path,'mask_binary_prf1_statics.txt'];

fids= fopen(savepath,'w');
fidr=fopen(filepath,'r');
recall=[];
precision=[];
f_measure=[];
k=1;
pattern='^(?<filename>\w*.bmp)\s*recall = (?<recall>\d*\.*\d+)\% precision = (?<precision>\d*\.*\d+)\% f_1 measure=(?<f1>\d*\.*\d+)\%$';
while ~feof(fidr)  
     tline=fgetl(fidr); 
     try
         os=regexp(tline,pattern,'names');
         r=str2double(os.recall);
         p=str2double(os.precision);
         f=str2double(os.f1);
         if(f==0)  %Consider if the mask is all black (corresponding to background image)
             disp(tline);
         else
              recall(k)=r;
              precision(k)=p;
              f_measure(k)=f;
              k=k+1;
         end
        
     catch
          disp(tline);
     end
end
recall_avg=mean(recall);
recall_std=std(recall);  
precision_avg=mean(precision);
precision_std=std(precision);

f_measure_avg=mean(f_measure);
f_measure_std=std(f_measure);

fprintf(fids,'recall_avg: %.2f%%     recall_std: %.2f%%\nprecision_avg: %.2f%%     precision_std: %.2f%%\nf_measure_avg: %.2f%%    f_measure_std: %.2f%%',recall_avg,...
    recall_std,precision_avg,precision_std,f_measure_avg,f_measure_std);

fclose(fids);
fclose(fidr);