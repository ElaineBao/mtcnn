clear all
clc

% load negative ImageNet
load('../imagenet/imagenet.mat');
load('imagenet_size.mat')
all_num = size(imagenet);
neg_root_path = '/ssd/zhaofan/IDCardDet_new/data/imagenet/image/';
[value,ids] = sort(min(size_table,[],2));
neg_num = 15000;
neg_list = imagenet(ids(end-neg_num+1:end));


load('all_back256_160.mat')

pos_num = 5000;
root_path = '/ssd/zhaofan/IDCardDet_new1/data/back/';
save_path = '/ssd/zhaofan/IDCardDet_new1/data/traintest/back_crop1/';
save_size = 256;

load('all_back256_160.mat');

back_txt = fopen('back_train_crop1.txt','w');

pos_ids = pos_num;
back_count = 0;


for i=1:neg_num
    
    if(mod(i,100)==0)
        disp(i)
    end
    if pos_ids < 1
        pos_ids = pos_num;
    end
    
    neg_img = imread([neg_root_path neg_list{i}]);
    [H,W,C] = size(neg_img);
    if(min(H,W) < save_size*1.7)
        continue;
    end
    if(1 == C)
        neg_img = repmat(neg_img,1,1,3);
    end
    
    r_h = randi([round(save_size*1.5) round(H-save_size*0.1)]);
    r_w = randi([round(save_size*1.5) round(W-save_size*0.1)]);
    r_x = randi([1, W - r_w]);
    r_y = randi([1, H - r_h]);
    neg_img = neg_img(r_y:r_y+r_h-1,r_x:r_x+r_w-1,:);
    sw = save_size * 2 / r_w;
    sh = save_size * 2 / r_h;
    neg_H = round(save_size * 2);
    neg_W = round(save_size * 2);
    neg_img = cwimresize(neg_img,[neg_H neg_W]);
    
    pos_H = randi([48 190]);
    pos_W = randi([96 200]);
    pos_img    =  all_back(:,:,:,pos_ids);
    pos_img = cwimresize(pos_img,[pos_H pos_W]);
    
    rx = randi([round(save_size*0.6) round(neg_W - pos_W - save_size*0.6)]);
    ry = randi([round(save_size*0.6) round(neg_H - pos_H - save_size*0.6)]);
    neg_img(ry:ry+pos_H-1,rx:rx+pos_W-1,:) = pos_img;
    
    
    
    r_angle = fix((rand(1)-0.5)*60);
    
    neg_img = imrotate(neg_img,-r_angle,'bilinear','crop');
    
    s_start = floor(save_size / 2);
    
    neg_img = neg_img(s_start:s_start+save_size-1,s_start:s_start+save_size-1,:);
    
    label = reshape(alllabel(pos_ids,:),2,4)';
    label(:,1) = label(:,1) * pos_W / 256 + rx;
    label(:,2) = label(:,2) * pos_H / 160 + ry;
    
    % rotation matrix
    r_mat = [cosd(r_angle),sind(r_angle);-sind(r_angle),cosd(r_angle)];
    label = (label - save_size) * r_mat + save_size - s_start;
  
    imwrite(neg_img,[save_path num2str(back_count) '.jpg']);
    offset = ([0 0; save_size 0; save_size save_size; 0 save_size] - label) / save_size;
    fprintf(back_txt,'%s %d %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n',[save_path num2str(back_count) '.jpg'],2,offset(:));
    
    img_old = neg_img;
    label_old = label;
    
    % rotate 90
    if(1)
        neg_img = imrotate(img_old,-90);
        label = (label_old - save_size / 2) * [0 1;-1,0] + save_size / 2;
        offset = ([0 0; save_size 0; save_size save_size; 0 save_size] - label) / save_size;
        imwrite(neg_img,[save_path num2str(back_count) '_1.jpg']);
        fprintf(back_txt,'%s %d %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n',[save_path num2str(back_count) '_1.jpg'],2,offset(:));
    end
    
    % rotate 180
    if(1)
        neg_img = imrotate(img_old,-180);
        label = (label_old - save_size / 2) * [-1 0;0,-1] + save_size / 2;
        offset = ([0 0; save_size 0; save_size save_size; 0 save_size] - label) / save_size;
        imwrite(neg_img,[save_path num2str(back_count) '_2.jpg']);
        fprintf(back_txt,'%s %d %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n',[save_path num2str(back_count) '_2.jpg'],2,offset(:));
    end
    
     % rotate 270
    if(1)
        neg_img = imrotate(img_old,-270);
        label = (label_old - save_size / 2) * [0 -1;1,0] + save_size / 2;
        offset = ([0 0; save_size 0; save_size save_size; 0 save_size] - label) / save_size;
        imwrite(neg_img,[save_path num2str(back_count) '_3.jpg']);
        fprintf(back_txt,'%s %d %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n',[save_path num2str(back_count) '_3.jpg'],2,offset(:));
    end
    
    
    pos_ids = pos_ids - 1;  
    back_count = back_count + 1;
    
    
    % debug
%     pHandle = figure;
%     imshow(neg_img)
%     hold on
%     r_pos = [0 0; save_size 0; save_size save_size; 0 save_size] - offset * save_size;
%     for p=1:4
%         hold on
%         plot(r_pos(p,1),r_pos(p,2),'o','LineWidth',4);
%         text(double(r_pos(p,1))+10,double(r_pos(p,2))+10,num2str(p),'FontSize',30,'Color','b')
%     end
%     pause
%     close(pHandle)
    
    
end


fclose(back_txt);
