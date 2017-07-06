clear all
clc
% load negative ImageNet
load('../imagenet/imagenet.mat');
load('imagenet_size.mat')
all_num = size(imagenet);
root_path = '/ssd/zhaofan/IDCardDet_new/data/imagenet/image/';
save_path = '/ssd/zhaofan/IDCardDet_new/data/traintest/neg/';
save_size = 256;

base_size = 375;



[value,ids] = sort(min(size_table,[],2));
index = find(value>=base_size);
index = index(1);

neg_txt = fopen('neg_train.txt','w');
neg_count = 0;

for i=index:all_num
    if (mod(i,1000) == 0)
        disp(i)
    end
    img = imread([root_path imagenet{ids(i)}]);
    img = cwimresize(img,[save_size save_size]);
    imwrite(img,[save_path num2str(neg_count) '.jpg']);
    fprintf(neg_txt,'%s %d\n',[save_path num2str(neg_count) '.jpg'],0);
    neg_count = neg_count + 1;
end
fclose(neg_txt);






