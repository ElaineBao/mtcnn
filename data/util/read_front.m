% read data after correction
clc
clear all

% read images and their labels

all_front   = [];
alllabel = [];
for i=1:4
    filepath = ['../front/1_' num2str(i) '_ImageFolder.txt'];
    mdata = importdata(filepath);
    all_front   = [all_front;  mdata.textdata]; %#ok<*AGROW>
    alllabel = [alllabel; mdata.data(:,2:end)];
end


%  checkout image path
for i=1:numel(all_front)
    mdata = all_front{i};
    mdata = mdata(4:end);
    mdata(4) = '/';
    all_front{i} = mdata; %#ok<*SAGROW>
end


% check out
img_num = numel(all_front);
errorlist = [];


for i=1:img_num
    if(mod(i,100)==0)
        disp(i)
    end
    img = imread(['../front/' all_front{i}]);
    [H,W,C] = size(img);
    if ( 1== C )
        img = repmat(img,1,1,3);
    end
    x1 = round(min(alllabel(i,1:2:8)));
    y1 = round(min(alllabel(i,2:2:8)));
    x2 = round(max(alllabel(i,1:2:8)));
    y2 = round(max(alllabel(i,2:2:8)));

    w  = x2 - x1+1;
    h  = y2 - y1+1;
    
    error_flag = 0;
    if (w>h && h>128)


        if((alllabel(i,5) < alllabel(i,1)) && (alllabel(i,7) > alllabel(i,3))...
                && (alllabel(i,6) < alllabel(i,2)) && (alllabel(i,8) < alllabel(i,4)))

        elseif (~((alllabel(i,5) > alllabel(i,1)) && (alllabel(i,7) < alllabel(i,3))...
                && (alllabel(i,6) > alllabel(i,2)) && (alllabel(i,8) > alllabel(i,4))))
            errorlist = [errorlist i];
            error_flag = 1;
        end
        
    elseif(w<h && w>128)
        if((alllabel(i,5) < alllabel(i,1)) && (alllabel(i,7) < alllabel(i,3))...
                && (alllabel(i,6) > alllabel(i,2)) && (alllabel(i,8) < alllabel(i,4)))
        elseif ((alllabel(i,5) > alllabel(i,1)) && (alllabel(i,7) > alllabel(i,3))...
                && (alllabel(i,6) < alllabel(i,2)) && (alllabel(i,8) > alllabel(i,4)))
        else
            errorlist = [errorlist i];
            error_flag = 1;
        end
    else
        errorlist = [errorlist i];
        error_flag = 1;
    end
    
    if(error_flag)
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

error_front = all_front(errorlist);
all_front(errorlist) = [];
alllabel(errorlist,:) = [];
save('all_front.mat','all_front','alllabel','errorlist');


