clear all
clc
load('all_back.mat')

train_num = 4900;
test_num  = 163;
root_path = '/ssd/zhaofan/IDCardDet_new/data/back/';
save_path = '/ssd/zhaofan/IDCardDet_new/data/traintest/back/';
save_size = 256;

back_txt = fopen('back_train.txt','w');

% train
for i=1:train_num
    if(mod(i,100) == 0)
        disp(i)
    end
    img = imread([root_path all_back{i}]);
    [H,W,C] = size(img);
    if (1 == C)
        img = repmat(img,1,1,3);
    end
    img = cwimresize(img,[save_size save_size]);
    imwrite(img,[save_path num2str(i) '.jpg']);
    label = alllabel(i,:);
    label = reshape(label, 2,4)';
    label(:,1) = label(:,1) * save_size / W;
    label(:,2) = label(:,2) * save_size / H;
    offset = ([0 0; save_size 0; save_size save_size; 0 save_size] - label) / save_size;
    fprintf(back_txt,'%s %d %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n',[save_path num2str(i) '.jpg'],2,offset(:));
    
    img_old = img;
    label_old = label;
    % rotate 90
    if(1)
        img = imrotate(img_old,-90);
        label = (label_old - save_size / 2) * [0 1;-1,0] + save_size / 2;
        offset = ([0 0; save_size 0; save_size save_size; 0 save_size] - label) / save_size;
        imwrite(img,[save_path num2str(i) '_1.jpg']);
        fprintf(back_txt,'%s %d %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n',[save_path num2str(i) '_1.jpg'],2,offset(:));
    end
    
%     % rotate 180
    if(1)
        img = imrotate(img_old,-180);
        label = (label_old - save_size / 2) * [-1 0;0,-1] + save_size / 2;
        offset = ([0 0; save_size 0; save_size save_size; 0 save_size] - label) / save_size;
        imwrite(img,[save_path num2str(i) '_2.jpg']);
        fprintf(back_txt,'%s %d %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n',[save_path num2str(i) '_2.jpg'],2,offset(:));
    end
%     
%     % rotate 270
    if(1)
        img = imrotate(img_old,-270);
        label = (label_old - save_size / 2) * [0 -1;1,0] + save_size / 2;
        offset = ([0 0; save_size 0; save_size save_size; 0 save_size] - label) / save_size;
        imwrite(img,[save_path num2str(i) '_3.jpg']);
        fprintf(back_txt,'%s %d %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n',[save_path num2str(i) '_3.jpg'],2,offset(:));
    end
    

    
    % debug
%     pHandle = figure;
%     imshow(img)
%     hold on
%     r_pos = [0 0; save_size 0; save_size save_size; 0 save_size] - offset * save_size;
%     for p=1:4
%         hold on
%         plot(r_pos(p,1),r_pos(p,2),'o','LineWidth',4);
%         text(double(r_pos(p,1))+10,double(r_pos(p,2))+10,num2str(p),'FontSize',30,'Color','b')
%     end
%     pause
%     close(pHandle)
%         
end
fclose(back_txt);




