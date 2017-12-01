import numpy as np

width = 1024
height = 384
x_scale = 1.0
y_scale = 1.0
E = 2.718281828

def transformBbox(a, d):
    cx = a[0][0] + d[0][0] * a[0][2] / x_scale
    cy = a[0][1] + d[0][1] * a[0][3] / y_scale
    if d[0][2] < 1:
        w = a[0][2] * np.exp(d[0][2]) / x_scale
    else:
        w= a[0][2] * d[0][2] * E / x_scale
    if d[0][3] < 1:
        h = a[0][3] * np.exp(d[0][3]) / y_scale
    else:
        h = a[0][3] *  d[0][3] * E / y_scale
    out_box = [[]]*4
    out_box[0] = min(max(cx-w/2, 0), width-1)
    out_box[1] = min(max(cy-h/2, 0), height-1)
    out_box[2] = max(min(cx+w/2, width-1), 0)
    out_box[3] = max(min(cy+h/2, height-1), 0)
    return out_box

delta_data = np.zeros(24)
anchor_data = np.zeros(24)
for i in range(0, 23):
    delta_data[i] = i * 0.1
    anchor_data[i] = i * 0.1

delta = np.reshape(delta_data, [1, 4, 6])
anchor = np.reshape(anchor_data, [1, 4, 6])

for i in range(0, 5):
    d = delta[:, :, i]
    a = anchor[:, :, i]
    print transformBbox(a, d)
