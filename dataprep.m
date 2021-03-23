% DATAPREP Performs data scaling
%
% Syntax
% [data, stat] = dataprep(data, method)
% where 
% data (matrix)   =  data matrix
% method (string) = is the name of method, 'std' or 'mms'
%                   'std' = variance scaling
%                   'mms' = min-max scaling
% stat (vector)   = statistic for post-processing
%
% 
% (c) Skön, J-P.; 8/10/2020; Savonia UAS 

function [data, stat] = dataprep(data, method)

  data = data';
  switch method
      case 'std'
        p = [mean(data,2), std(data,0,2)];
        p(find(abs(p(:,2)) < eps),2) = 1;
        data = norm_do(data, p);
      case 'mms'  
        mi = min(data,[],2); 
        ma = max(data,[],2);
        p = [mi + 0.5*(ma-mi), 0.5*(ma-mi)];
        p(find(abs(p(:,2)) < eps),2) = 1;
        data = norm_do(data, p);
      otherwise
        error('method not found')
  end
  data = data';
  stat = p;
  
function data = norm_do(data, p)

  dim = size(data,2);
  data = (data - repmat(p(:,1),1,dim)) ./ repmat(p(:,2),1,dim);