from os import listdir
import cv2
import argparse

def crop_img(img, size = (448, 640)):
    th, tw = size
    h, w = img.shape[0], img.shape[1]
    if w == tw and h == th:
        return img

    # resize the image if the image size is smaller than the target size
    scale_h, scale_w, scale = 1., 1., 1.
    if th > h:
        scale_h = float(th)/h
    if tw > w:
        scale_w = float(tw)/w
    if scale_h>1 or scale_w>1:
        scale = max(scale_h, scale_w)
        w = int(round(w * scale)) # w after resize
        h = int(round(h * scale)) # h after resize

    x1 = int((w-tw)/2)
    y1 = int((h-th)/2)

    if len(img.shape)==3:
        if scale>1:
            img = cv2.resize(img, (w,h), interpolation=cv2.INTER_LINEAR)
        cropped = img[y1:y1+th,x1:x1+tw,:]
    elif len(img.shape)==2:
        if scale>1:
            img = cv2.resize(img, (w,h), interpolation=cv2.INTER_LINEAR)
        cropped = img[y1:y1+th,x1:x1+tw]

    return cropped

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='HRL')
    parser.add_argument('--img-dir', default='',
                        help='image folder where the RGB or depth images are (default: "")')
    args = parser.parse_args()

    # imgfolder = 'data/fr3_walking_xyz/associated/depth'
    imgfolder = args.img_dir
    file_type = imgfolder.split("/")[-1] if imgfolder[-1] != "/" else imgfolder.split("/")[-2]
    files = listdir(imgfolder)
    rgbfiles = [(imgfolder +'/'+ ff) for ff in files if (ff.endswith('.png') or ff.endswith('.jpg'))]
    rgbfiles.sort()
    imgfolder = imgfolder

    print('Find {} image files in {}'.format(len(rgbfiles), imgfolder))

    for i in range(len(rgbfiles)):
        imgfile = rgbfiles[i].strip()
        img = cv2.imread(imgfile, cv2.IMREAD_UNCHANGED)
        cropped = crop_img(img)
        cv2.imwrite('../tartanvo/data/fr3_walking_xyz/cropped/' + file_type + '/%04d.png' % (i), cropped)