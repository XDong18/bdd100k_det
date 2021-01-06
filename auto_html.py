import os

img_dir = 'show/mask_rcnn_r_101_fpn_yanzaho'
img_list = sorted(os.listdir(img_dir))
out_fn = 'show/mask_rcnn_r_101_fpn_yanzaho.html'
with open (out_fn, 'w') as f:
    f.write('<html>\n')
    f.write('<body>\n')
    f.write('<h1>mask_rcnn_r_101_fpn_yanzaho</h1>\n')

    for i, img in enumerate(img_list):
        name = "<p>" + str(i) + ' ---- ' + img + "</p>\n"
        f.write(name)
        new_line = "<img src='" + os.path.join('mask_rcnn_r_101_fpn_yanzaho', img) + "'>\n"
        f.write(new_line)

    # f.write("<img src='" + "test.jpg" + "'>\n")
    f.write('</body>\n')
    f.write('</html>\n')
