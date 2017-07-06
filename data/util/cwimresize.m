function imgB = cwimresize( imgA,new_size)

[h,w,~] = size(imgA);
new_h = new_size(1);
new_w = new_size(2);

sh = new_h / h;
sw = new_w / w;
max_scale = max(sw,sh);

if (min(sw,sh)>2)
    imgB = imresize(imgA,new_size,'nearest');
elseif(max_scale>0.5)
    imgB = imresize(imgA,new_size,'bilinear','antialiasing',false);
else
    while (max_scale <= 0.25)
        imgA = imresize(imgA,0.25,'bilinear');
        max_scale = max_scale * 4;
    end
    while (max_scale <= 0.5)
        imgA = imresize(imgA,0.5,'bilinear');
        max_scale = max_scale * 2;
    end
    imgB = imresize(imgA,new_size,'bilinear','antialiasing',false);
        
end

end

