from PIL import Image
import numpy
import pylab
from scipy.ndimage import label

def crop_it(seg, card):
    rows = numpy.sum(card, axis=1)
    cols = numpy.sum(card, axis=0)
    rinds = numpy.where(rows != 0)
    cinds = numpy.where(cols != 0)
##    print(rinds[0], rinds[-1])
    return seg[rinds[0][0]:rinds[0][-1],cinds[0][0]:cinds[0][-1]]

def card_segmentation(img):
##    for i in range(img.shape[-1]):
##        pylab.figure()
##        pylab.hist(img[:,:,i].ravel(),bins=range(256),log=True)
##        pylab.show()

    seg = img[:,:,1] > 130

    cards,nlb = label(seg)
    hists, lbs = numpy.histogram(cards,bins=nlb)
    inds = numpy.argsort(hists)[::-1][1:]

    fig, axes = pylab.subplots(4,3)
    for i in range(12):
        card = crop_it(img, cards == inds[i])
        
        r,c = divmod(i,3)
        axes[r][c].imshow(card)
    pylab.show()

if __name__ == "__main__":
    import os
    pathname = "..\set纸牌2"
    for r, d, fs in os.walk(pathname):
        for f in fs:
            fname = os.path.join(r,f)
            print(fname)
            if os.path.splitext(fname)[-1].upper() == ".PNG":
                img = numpy.array(Image.open(fname))#"IMG_8.png")
                card_segmentation(img)#[:,150:])
