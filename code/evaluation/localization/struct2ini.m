function struct2ini(filename,iris)
%==========================================================================
% Author:      Dirk Lohse ( dirklohse@web.de )
% Version:     0.1a
% Last change: 2008-11-13
%==========================================================================
%
% struct2ini converts a given structure into an ini-file.
% It's the opposite to Andriy Nych's ini2struct. Only 
% creating an ini-file is implemented. To modify an existing
% file load it with ini2struct.m from:
%       Andriy Nych ( nych.andriy@gmail.com )
% change the structure and write it with struct2ini.
%

% Open file, or create new file, for writing
% discard existing contents, if any.
fid = fopen(filename,'w'); 
fprintf(fid,'[iris]\n'); 
if(iris(1).exist)
  fprintf(fid,'exist=true\n'); 
   fprintf(fid,'center_x=%.2f\n', iris(1).center_x); 
   fprintf(fid,'center_y=%.2f\n',iris(1).center_y); 
   fprintf(fid,'radius=%.2f\n',iris(1).radius); 
else
  fprintf(fid,'exist=false');   
end
fprintf(fid,'\n');
fprintf(fid,'[pupil]\n'); 
if(iris(2).exist)
  fprintf(fid,'exist=true\n'); 
   fprintf(fid,'center_x=%.2f\n', iris(2).center_x); 
   fprintf(fid,'center_y=%.2f\n',iris(2).center_y); 
   fprintf(fid,'radius=%.2f\n',iris(2).radius); 
else
  fprintf(fid,'exist=false');   
end

fclose(fid); % close file