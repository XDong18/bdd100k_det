import os

img_dir = 'show/mask_rcnn_R_50_FPN_3x_val'
img_list = os.listdir(img_dir)
out_fn = 'html/mask_rcnn_R_50_FPN_3x_val.html'
with open (out_fn, 'w') as f:
    f.write('<html>\n')
    f.write('<body>\n')
    f.write('<h1>mask_rcnn_R_50_FPN_3x_val</h1>\n')

    for img in img_list:
        name = "<p>" + img + "</p>\n"
        f.write(name)
        new_line = "<img src='" + os.path.join(img_dir, img) + "'>\n"
        f.write(new_line)

    # f.write("<img src='" + "test.jpg" + "'>\n")
    f.write('</body>\n')
    f.write('</html>\n')
