"""Half-open interval algebra; duration sums must never double count streams."""
import math


def union_intervals(intervals):
    rows=[]
    for start,end in intervals:
        if not math.isfinite(start) or not math.isfinite(end) or end<start:
            raise ValueError('Invalid interval')
        if end>start:rows.append((start,end))
    merged=[]
    for start,end in sorted(rows):
        if merged and start<=merged[-1][1]:merged[-1]=(merged[-1][0],max(end,merged[-1][1]))
        else:merged.append((start,end))
    return merged


def intersect_intervals(left,right):
    left,right=union_intervals(left),union_intervals(right)
    output=[];i=j=0
    while i<len(left) and j<len(right):
        start,end=max(left[i][0],right[j][0]),min(left[i][1],right[j][1])
        if end>start:output.append((start,end))
        if left[i][1]<right[j][1]:i+=1
        else:j+=1
    return output


def interval_duration(intervals):
    return sum(end-start for start,end in union_intervals(intervals))


def difference_duration(left,right):
    return interval_duration(left)-interval_duration(intersect_intervals(left,right))
