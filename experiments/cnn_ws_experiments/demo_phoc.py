# -*- coding: utf-8 -*-
import torch
from collections import defaultdict
import string
import cv2
import pymsgbox #pip install PyMsgBox
import glob
import cnn_ws.string_embeddings.phoc
from cnn_ws.models.myphocnet import PHOCNet
from cnn_ws_experiments.datasets.gw_alt import GWDataset
import cnn_ws.utils.save_load
import numpy as np
import tqdm
import pickle
import Tkinter
from PIL import Image
from PIL import ImageTk
import tkFileDialog
import tkSimpleDialog
import cv2
import sklearn.neighbors
import scipy.spatial.distance
letter_mapper=defaultdict(lambda:'')
letter_mapper.update({'á':'a', 'é':'e', 'í':'i','ö':'o', 'ú':'u','ü':'u','ý':'y', 'č':'c','ď':'d','ě':'e',
          'ň':'n', 'ř':'r', 'š':'s', 'ť':'t', 'ů':'u', 'ž':'z'})
letter_mapper.update({k: k for k in string.lowercase+string.digits})
stem_caption=lambda x:''.join([letter_mapper[l] for l in x.lower()])
import time


def dtp(img, column_range=list(range(1, 40, 3)), row_range=list(range(1, 40, 3)), threshold_coefficients=[0.7, 1.01, 0.1]):
    def unique_boxes(boxes, downscale=1):
        scaled_box = boxes.astype("int64") / downscale
        assert scaled_box.max() < 2 ** 15
        int_boxes = scaled_box[:, 0] + scaled_box[:, 1] * (2 ** 15) + scaled_box[:, 2] * (2 ** 31) + scaled_box[:,
                                                                                                     3] * (2 ** 47)
        _, index = np.unique(int_boxes, return_index=True)
        return boxes[index, :]

    def filter_boxes(boxes):
        w = boxes[:,2] - boxes[:,0]
        h = boxes[:, 2] - boxes[:, 0]
        keep = w>h
        int_boxes = scaled_box[:, 0] + scaled_box[:, 1] * (2 ** 15) + scaled_box[:, 2] * (2 ** 31) + scaled_box[:,
                                                                                                     3] * (2 ** 47)
        _, index = np.unique(int_boxes, return_index=True)
        return boxes[index, :]


    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_mean = img.mean()
    all_boxes = []
    for threshold_coefficient in threshold_coefficients:
        binary_image = ((img < img_mean * threshold_coefficient).astype(np.ubyte))
        for R in row_range:
            for C in column_range:
                s_img = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, np.ones((R, C), dtype=np.ubyte))
                n, l_img, stats, centroids = cv2.connectedComponentsWithStats(s_img, connectivity=4)
                box_list = [[b[0], b[1], b[0] + b[2], b[1] + b[3]] for b in stats]
                all_boxes += box_list
    all_boxes = np.array(all_boxes)
    all_boxes = unique_boxes(all_boxes)
    return all_boxes


def plot_rects(img, rects, is_ltrb=True,margin=1,safety_margin=5):
    sz=[n+safety_margin for n in img.shape]
    if is_ltrb:
        ltrb=rects#[:,[1,0,3,2]]
    else:
        ltrb=np.zeros_like(rects)
        ltrb[:,:2]=rects[:,:2]
        ltrb[:, 2:] = rects[:, :2]+rects[:, 2:]
        #ltrb[:,:]=ltrb[:,[1,0,3,2]]

    #idx=((ltrb[:,0]-ltrb[:,2])**2>4)+((ltrb[:,1]-ltrb[:,3])**2>4)==0
    print(ltrb.max(axis=0))
    ltrb=ltrb.astype("int32")

    bglarge=np.zeros(sz)
    print(ltrb[:,0])
    np.add.at(bglarge,(ltrb[:,1],ltrb[:,0]),1)
    np.add.at(bglarge, (ltrb[:, 3], ltrb[:, 2]), 1)
    np.add.at(bglarge,(ltrb[:,1],ltrb[:,2]),-1)
    np.add.at(bglarge, (ltrb[:, 3], ltrb[:, 0]), -1)
    bglarge=bglarge.cumsum(axis=0).cumsum(axis=1)


    ltrb[:, 0] += margin
    ltrb[:, 1] += margin
    ltrb[:, 2] -= margin
    ltrb[:, 3] -= margin
    bgsmall=np.zeros(sz)

    np.add.at(bgsmall,(ltrb[:,1],ltrb[:,0]),1)
    np.add.at(bgsmall, (ltrb[:, 3], ltrb[:, 2]), 1)
    np.add.at(bgsmall,(ltrb[:,1],ltrb[:,2]),-1)
    np.add.at(bgsmall, (ltrb[:, 3], ltrb[:, 0]), -1)

    bgsmall=bgsmall.cumsum(axis=0).cumsum(axis=1)
    res = bglarge-bgsmall
    res = res[:-safety_margin,:-safety_margin]
    return res, bglarge



def get_query_phoc(query,ds):
    unigrams = [chr(i) for i in range(ord('a'), ord('z') + 1) + range(ord('0'), ord('9') + 1)]
    phoc_unigram_levels = (1, 2, 4, 8)
    word_strings = [query]+[elem[1] for elem in ds.words]
    return cnn_ws.string_embeddings.phoc.build_phoc_descriptor(word_strings, unigrams, phoc_unigram_levels)[0]

class WsDemo():
    def load_dataset(self,root_dir,cnn_fname=None):
        self.ds = GWDataset(gw_root_dir=root_dir,image_extension='.png',min_image_width_height=26,cv_split_method="almazan")
        self.ds.mainLoader()
        self.cnn = PHOCNet(n_out=self.ds[0][1].shape[0],input_channels=1,gpp_type='gpp',pooling_levels=([1], [5]))
        if cnn_fname is not None:
            self.cnn.load_state_dict(torch.load(cnn_fname, map_location=lambda storage, loc: storage))
        #pattern=root_dir+"/*/*.png"
        #img_files=glob.glob(pattern)
        #gt_files = [f[:-3].replace("pages","ground_truth")+"gtp" for f in img_files]
        #print "Loading {} images ... ".format(len(img_files)),
        #self.images = {f.split("/")[-1].split(".")[0]:cv2.imread(f) for f in img_files}
        #print "done."

        #self.small_images = {k: cv2.resize(v, (0, 0), fx=.3, fy=.3) for k,v in self.images.items()}
        #self.gray_images = {k: cv2.cvtColor(v, cv2.COLOR_BGR2GRAY) for k, v in self.images.items()}


    def get_embeddings(self,fname):
        try:
            self.embeddings=pickle.load(open(fname,"rb"))
        except:
            self.embeddings=np.zeros([len(self.ds),self.ds[0][1].shape[0]])
            with torch.no_grad():
                for n, data in enumerate(tqdm.tqdm(torch.utils.data.DataLoader(self.ds))):
                    self.embeddings[n, :]=torch.sigmoid(self.cnn(data[0])).cpu().numpy()
            pickle.dump(self.embeddings, open(fname, "wb"))
        #self.knn=sklearn.neighbors.NearestNeighbors(n_neighbors=200)
        #self.knn.fit(self.embeddings,np.arange(len(self.embeddings)))

    def qbe(self,img):
        with torch.no_grad():
            img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            img= 1 - img.astype(np.float32) / 255.0
            img=torch.Tensor(img).unsqueeze(0).unsqueeze(0)
            query=torch.sigmoid(self.cnn(img)).cpu().numpy()
            resp=np.argsort(scipy.spatial.distance.cdist(self.embeddings[:, :], query,metric='cosine').reshape(-1))
        return [1-self.ds[n][0].numpy()[0,:,:] for n in resp]


    def qbs(self,stri):
        stri=stem_caption(stri.lower())
        query=get_query_phoc(stri,self.ds)
        query=query.reshape(1,-1)
        resp = np.argsort(scipy.spatial.distance.cdist(self.embeddings[:, :], query, metric='cosine').reshape(-1))
        return [1 - self.ds[n][0].numpy()[0, :, :] for n in resp]


    def __init__(self):
        self.load_dataset(root_dir="./data/cb/",cnn_fname="PHOCNet.cb.pt")
        self.get_embeddings("cb_embeddings.pickle.bkup")



if __name__=="__main__":
    db=WsDemo()
    def plot_results(image_list,title,item_height,page_wh,padding=1):
        top=padding
        left=padding
        res = np.zeros([page_wh[1],page_wh[0],3])
        for img in image_list:
            scale = float(item_height) / img.shape[0]
            #print scale,img.shape,(item_height,int(img.shape[1] * scale))
            img_small = cv2.resize(img, (int(img.shape[1] * scale),item_height))
            if len(img_small.shape)==2:
                img_small=cv2.cvtColor(img_small,cv2.COLOR_GRAY2BGR)

            if left+padding+img_small.shape[1] < page_wh[0]:
                res[top:top+img_small.shape[0], left:left+img_small.shape[1],:]=img_small
                left+=img_small.shape[1]+padding
            elif 2*padding+img_small.shape[1] > page_wh[0]: # word biger than our window
                print "Large word aborting!"
                break
            elif top+padding+2*img_small.shape[0] < page_wh[1]:
                left = padding
                top += padding + img_small.shape[0]
                res[top:top + img_small.shape[0],left:left + img_small.shape[1], :] = img_small
                left += img_small.shape[1] + padding
            else: # below bottom
                break
        cv2.namedWindow(title)
        cv2.moveWindow(title,400,400)
        #cv2.resizeWindow("Query Image", 400, 50)
        cv2.imshow(title,res)

    def qbe():
        cv2.destroyAllWindows()
        path = tkFileDialog.askopenfilename()
        img=cv2.imread(path)

        scale=950.0/img.shape[0]

        img_small=cv2.resize(img,(int(img.shape[1]*scale),int(img.shape[0]*scale)))
        cv2.namedWindow("Select A query")
        roi=cv2.selectROI("Select A query", img_small, False, False)
        cv2.destroyWindow("Select A query")
        l,t,r,b=[int(n/scale) for n in roi]
        b+=t
        r+=l
        query_img=img[t:b,l:r]
        cv2.namedWindow("Query Image")
        cv2.moveWindow("Query Image",400,50)
        cv2.imshow("Query Image",query_img)
        set_status("QBE Searching Query...")
        t=time.time()
        res=db.qbe(query_img)
        plot_results(res,"QBE" ,50,(1000,500))
        set_status("QBE Retrieval in {:03f} sec.".format(time.time()-t))
        cv2.waitKey(10)

    def qbs():
        # grab a reference to the image panels
        cv2.destroyAllWindows()
        query_string=tkSimpleDialog.askstring("Query Entry", "Which word are you searching for")
        set_status("QBS ({})\nSearching Query...".format(query_string))
        t=time.time()
        res=db.qbs(query_string)
        plot_results(res,"QBS: '{}'".format(query_string),50,(1000,500))
        set_status("QBS Retrieval in {:03f} sec.".format(time.time() - t))
        cv2.waitKey(10)
    def load_model():
        cv2.destroyAllWindows()
        cnn_fname = tkFileDialog.askopenfilename()
        print "Loading {} ...".format(cnn_fname),
        db.cnn.load_state_dict(torch.load(cnn_fname, map_location=lambda storage, loc: storage))
        print "done."
        set_status("Loaded Model: {}".format(cnn_fname.split("/")[-1]))

    def show_dtp():
        cv2.destroyAllWindows()
        path = tkFileDialog.askopenfilename()
        img=cv2.imread(path)
        scale=int(tkSimpleDialog.askinteger("Percentage", "To how much should the image be scaled."))/100.0
        keep=int(tkSimpleDialog.askinteger("Percentage", "How mamy proposals should we show."))/100.0
        img_small = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))
        set_status("Computing proposals")
        t=time.time()
        proposals=dtp(cv2.cvtColor(img_small,cv2.COLOR_BGR2GRAY))
        scale=700.0/img.shape[0]
        proposals=proposals[np.random.rand(len(proposals))<keep,:]
        prop_img,_=plot_rects(np.zeros([img_small.shape[0],img_small.shape[1]]),proposals)
        print 'Before:',img_small.shape
        scale=700.0/img_small.shape[0]
        img_small = cv2.resize(img_small, (int(img_small.shape[1] * scale), int(img_small.shape[0] * scale)))
        prop_img = cv2.resize(prop_img, (int(prop_img.shape[1] * scale), int(prop_img.shape[0] * scale)))
        print 'After:',img_small.shape
        for n in range(3):
            img_small[:,:,n]=img_small[:,:,n]*(1-prop_img)
        #cv2.namedWindow("Proposals")
        cv2.imshow("Proposals", img_small)
        cv2.imwrite("proposals.png",img_small)
        cv2.waitKey(100)
        set_status("{} Proposals computed\nTime {:03f} sec.\nShowing {} proposals".format(len(proposals),time.time()-t,int(len(proposals)*keep/100.0)))


    win_root = Tkinter.Tk()


    # create a button, then when pressed, will trigger a file chooser
    # dialog and allow the user to select an input image; then add the
    # button the GUI

    msg = Tkinter.Message(win_root, text=""*4,width = 200, border=1)
    msg.pack(side="bottom",fill="both", expand="yes", padx="10", pady="10")

    def set_status(status):
        msg.config(text=status, width=200)
        msg.update_idletasks()

    btn_model = Tkinter.Button(win_root, text="Load Model", command=load_model)
    btn_model.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")
    btn_qbe = Tkinter.Button(win_root, text="Query By Example", command=qbe)
    btn_qbe.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")
    btn_qbs = Tkinter.Button(win_root, text="Query By String", command=qbs)
    btn_qbs.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")
    btn_dtp = Tkinter.Button(win_root, text="Proposals", command=show_dtp)
    btn_dtp.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")

    # kick off the GUI
    win_root.mainloop()
