clear all
clc
% load negative ImageNet
load('../imagenet/imagenet.mat');
neg_root = '/ssd/zhaofan/IDCardDet_new/data/imagenet/image/';

num = numel(imagenet);

size_table = zeros(num,2);

for i=1:num
    if(mod(i,1000) == 0)
        disp(i)
    end
    img = imread([neg_root imagenet{i}]);    
    [M,N,C] = size(img);    
    size_table(i,:) = [M,N];
    
end

save('imagenet_size.mat','size_table')
