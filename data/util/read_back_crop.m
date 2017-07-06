% read data after correction
clc
clear all

% read images and their labels

allimg   = [];
alllabel = [];
for i=1:4
    filepath = ['../back/1_' num2str(i) '_ImageFolder.txt'];
    mdata = importdata(filepath);
    allimg   = [allimg;  mdata.textdata]; %#ok<*AGROW>
    alllabel = [alllabel; mdata.data(:,2:end)];
end


%  checkout image path
for i=1:numel(allimg)
    mdata = allimg{i};
    mdata = mdata(4:end);
    mdata(4) = '/';
    allimg{i} = mdata; %#ok<*SAGROW>
end


% check out
img_num = numel(allimg);
save_w = 256;
save_h = 160;
all_back = zeros(save_h,save_w,3,img_num,'uint8');
base_points = [];
errorlist = [];


for i=1:img_num
    if(mod(i,100)==0)
        disp(i)
    end
    img = imread(['../back/' allimg{i}]);
    [H,W,C] = size(img);
    if ( 1== C )
        img = repmat(img,1,1,3);
    end
    x1 = round(min(alllabel(i,1:2:8)));
    y1 = round(min(alllabel(i,2:2:8)));
    x2 = round(max(alllabel(i,1:2:8)));
    y2 = round(max(alllabel(i,2:2:8)));
    alllabel(i,1:2:8) = alllabel(i,1:2:8)-x1;
    alllabel(i,2:2:8) = alllabel(i,2:2:8)-y1;
    w  = x2 - x1+1;
    h  = y2 - y1+1;
    img_temp = zeros(h,w,3,'uint8');
    offset_x = 1;
    if(x1<1)
        offset_x = 1-x1;
        x1 = 1;
    end
    offset_y = 1;
    if(y1<1)
        offset_y = 1-y1;
        y1 = 1;
    end
    x2 = min(x2,W);
    y2 = min(y2,H);
    w  = x2 - x1;
    h  = y2 - y1;
    img_temp(offset_y:offset_y+h,offset_x:offset_x+w,:) = img(y1:y2,x1:x2,:);
    
    error_flag = 0;
    if (w>h && h>64)
        img_temp = cwimresize(img_temp,[save_h,save_w]);
        alllabel(i,1:2:8) = alllabel(i,1:2:8)*save_w/w;
        alllabel(i,2:2:8) = alllabel(i,2:2:8)*save_h/h;
        if((alllabel(i,5) < alllabel(i,1)) && (alllabel(i,7) > alllabel(i,3))...
                && (alllabel(i,6) < alllabel(i,2)) && (alllabel(i,8) < alllabel(i,4)))
            img_temp = imrotate(img_temp,180);
            alllabel(i,1:2:8) = save_w - alllabel(i,1:2:8);
            alllabel(i,2:2:8) = save_h  - alllabel(i,2:2:8);
        elseif (~((alllabel(i,5) > alllabel(i,1)) && (alllabel(i,7) < alllabel(i,3))...
                && (alllabel(i,6) > alllabel(i,2)) && (alllabel(i,8) > alllabel(i,4))))
            errorlist = [errorlist i];
            error_flag = 1;
        end
        
    elseif(w<h && w>64)
        img_temp = cwimresize(img_temp,[save_w,save_h]);
        alllabel(i,1:2:8) = alllabel(i,1:2:8)*save_h/w;
        alllabel(i,2:2:8) = alllabel(i,2:2:8)*save_w/h;
        if((alllabel(i,5) < alllabel(i,1)) && (alllabel(i,7) < alllabel(i,3))...
                && (alllabel(i,6) > alllabel(i,2)) && (alllabel(i,8) < alllabel(i,4)))
            img_temp = imrotate(img_temp,90);
            x_temp = alllabel(i,1:2:8);
            alllabel(i,1:2:8) = alllabel(i,2:2:8);
            alllabel(i,2:2:8) = save_h - x_temp;

        elseif ((alllabel(i,5) > alllabel(i,1)) && (alllabel(i,7) > alllabel(i,3))...
                && (alllabel(i,6) < alllabel(i,2)) && (alllabel(i,8) > alllabel(i,4)))
            img_temp = imrotate(img_temp,-90);
            y_temp = alllabel(i,2:2:8);
            alllabel(i,2:2:8) = alllabel(i,1:2:8);
            alllabel(i,1:2:8) = save_w - y_temp;
        else
            errorlist = [errorlist i];
            error_flag = 1;
        end
    else
        errorlist = [errorlist i];
        error_flag = 1;
    end
    
    if(~error_flag)
        all_back(:,:,:,i) = img_temp;
    else
        fprintf('%s %d\n','Error: ',i);
    end
    
    %debug
%     pHandle = figure;
%     imshow(img_temp);
%     title(num2str(i));
%     hold on
%     xlim([-10 save_w+20])
%     ylim([-10 save_w+10])
%     for p=1:4
%         hold on
%         x = alllabel(i,(p-1)*2+1);
%         y = alllabel(i,p*2);
%         plot(x,y,'o','LineWidth',4);
%         text(double(x)+5,double(y)+5,num2str(p),'FontSize',30,'Color','b')
%         hold on
%     end
%     pause
%     close(pHandle);
    
end

all_back(:,:,:,errorlist) = [];
alllabel(errorlist,:) = [];
errorids  = errorlist;
errorlist = allimg(errorlist);
save('all_back256_160.mat','all_back','alllabel','errorids','errorlist');


