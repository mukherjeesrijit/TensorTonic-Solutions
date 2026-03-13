def iou(box_a, box_b):
    """
    Compute Intersection over Union of two bounding boxes.
    """

    x1,y1,x2,y2 = box_a
    a1,b1,a2,b2 = box_b
    top_x, top_y = max(x1,a1), max(y1,b1)
    bot_x, bot_y = min(x2,a2), min(y2,b2)
    len_x = max(0,bot_x-top_x) 
    len_y = max(0,bot_y-top_y)
    intersect = len_x * len_y
    print(intersect)
    union = (y2-y1)*(x2-x1) + (b2-b1)*(a2-a1) - intersect
    print(union)
    return intersect/union
    