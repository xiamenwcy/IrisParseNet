function hausdorffDistance = Hausdorff(S,G)
% Hausdorff calculates hausdorff distance between segmented objects in S
% and ground truth objects in G
%
% Inputs:
%   S: a label image contains segmented objects
%   G: a label image contains ground truth objects
%
% Outputs:
%   hausdorffDistance: as the name indicated
%
% Korsuk Sirinukunwattana
% BIAlab, Department of Computer Science, University of Warwick
% 2015
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% convert S and G to the same format
S = single(S);
G = single(G);

% check if S or G is non-empty

listS = unique(S);            % list of labels of segmented objects
listS(listS == 0) = [];       % remove the label of the background
numS = length(listS);         % the total number of segmented objects in S

listG = unique(G);            % list of labels of ground truth objects
listG(listG == 0) = [];       % remove the label of the background
numG = length(listG);         % the total number of ground truth in G

if numS == 0 && numG == 0    % no segmented object & no ground truth objects
    hausdorffDistance = 0;
    return
elseif numS == 0 || numG == 0
    hausdorffDistance = Inf;
    return
else
    % do nothing
end

% Calculate Hausdorff distance
maskS = S > 0;
maskG = G > 0;
[rowInd,colInd] = ind2sub(size(S),1:numel(S));
coordinates = [rowInd',colInd'];

x = coordinates(maskG,:);
y = coordinates(maskS,:);

% sup_{x \in G} inf_{y \in S} \|x-y\|
[~,dist] = knnsearch(y,x);
dist1 = max(dist);

% sup_{x \in S} inf_{y \in G} \|x-y\|
[~,dist] = knnsearch(x,y);
dist2 = max(dist);

hausdorffDistance = max(dist1,dist2);

end