from collections import defaultdict
import json
import glob
import string
import cv2

scale=1.0

letter_mapper=defaultdict(lambda:'')
letter_mapper.update({'á':'a', 'é':'e', 'í':'i','ö':'o', 'ú':'u','ü':'u','ý':'y', 'č':'c','ď':'d','ě':'e',
          'ň':'n', 'ř':'r', 'š':'s', 'ť':'t', 'ů':'u', 'ž':'z'})
letter_mapper.update({k: k for k in string.ascii_lowercase+string.digits})

stem_caption=lambda x:''.join([letter_mapper[l] for l in x])

all=0
kept=0
for n,file in enumerate(glob.glob("*/*.json")):
    new_id="pages/{:010}".format(n)
    imagefile=file[:-4]+"jpg"
    print(imagefile)
    img=cv2.imread(imagefile,cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
    if img is None:
        sys.exit()
    cv2.imwrite("pages/{:010d}.png".format(n),img)
    d=json.load(open(file))
    ltrb = d['rectangles_ltrb']
    ltrb = [[int(n*scale) for n in row] for row in ltrb]
    captions = d['captions']
    captions=[stem_caption(c) for c in captions]
    all+=len(captions)
    lines=["{} {} {} {} {}".format(*(ltrb[n]+[captions[n]])) for n in range(len(captions)) if len(captions[n])>3]
    kept+=len(lines)
    open("ground_truth/{:010d}.gtp".format(n), "w").write("\n".join(lines))

print("All words:",all)
print("Kept words:",kept)
