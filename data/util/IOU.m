function ovr = IOU(box,boxes)
% Compute IoU between detect box and gt boxes
% box: shape (1,5): x1, y1, x2, y2, score input box
% boxes: shape (n, 4): x1, y1, x2, y2,ground truth boxes
% ovr: shape (n,1), IoU

    box_area = (box(3) - box(1)+ 1) * (box(4) - box(2) + 1);
    area = (boxes(:, 3) - boxes(:, 1) + 1) .* (boxes(:, 4) - boxes(:, 2) + 1);
    xx1 = max(boxes(:, 1),box(1));
    yy1 = max(boxes(:, 2),box(2));
    xx2 = min(boxes(:, 3),box(3));
    yy2 = min(boxes(:, 4),box(4));

    % compute the width and height of the bounding box
    w = max(xx2 - xx1 + 1,0);
    h = max(yy2 - yy1 + 1,0);

    inter = w .* h;
    ovr = inter ./ (box_area + area - inter);
end
