function [tp,fp,tn,fn]=compute_image(msk,gt)
   interaction =and(msk,gt);
   union = or(msk,gt);
   g_left=xor(gt,interaction);
   msk_left=xor(msk,interaction);
   
   tp=nnz(interaction)/numel(msk);
   tn=nnz(~union)/numel(msk);
   fn=nnz(g_left)/numel(msk);
   fp=nnz(msk_left)/numel(msk); 
  
 