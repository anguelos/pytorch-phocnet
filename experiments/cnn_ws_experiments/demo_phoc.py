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



def get_query_phoc(query):
    unigrams = [chr(i) for i in range(ord('a'), ord('z') + 1) + range(ord('0'), ord('9') + 1)]
    phoc_unigram_levels = (1, 2, 4, 8)
    cnn_ws.string_embeddings.phoc.build_phoc_descriptor([[query]], unigrams, phoc_unigram_levels)

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
            print "Computing db_embeddings ... ",
            self.embeddings=np.zeros([len(self.ds),self.ds[0][1].shape[0]])
            with torch.no_grad():
                for n, data in enumerate(tqdm.tqdm(torch.utils.data.DataLoader(self.ds))):
                    self.embeddings[n, :]=torch.sigmoid(self.cnn(data[0])).cpu().numpy()
            print "done."
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
            print query
            #_, resp=self.knn.kneighbors(query.reshape(1,-1))
            #resp=resp.astype("uint32").reshape(-1)
            print resp
        return [1-self.ds[n][0].numpy()[0,:,:] for n in resp]


    def qbs(self,stri):
        stri=stem_caption(stri.lower())
        query=get_query_phoc(stri)
        _, resp=self.knn.kneighbors(query.reshape(1,-1))
        resp = resp.astype("uint32").reshape(-1)
        return [1-self.ds[n][0].numpy()[0,:,:] for n in resp]

    def __init__(self):
        self.load_dataset(root_dir="./data/cb/",cnn_fname="PHOCNet.cb.pt")
        self.get_embeddings("cb_embeddings.pickle.bkup")



if __name__=="__main__":
    db=WsDemo()
    def plot_results(image_list,item_height,page_wh,padding=1):
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
                print "Filled all!"
                break
        cv2.namedWindow("Query Results")
        cv2.moveWindow("Query Results",400,400)
        #cv2.resizeWindow("Query Image", 400, 50)
        cv2.imshow("Query Results",res)

    def qbe():
        print("QBE")
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
        print "ROI:",l,t,r,b
        query_img=img[t:b,l:r]
        print "Query.shape:",query_img.shape
        cv2.namedWindow("Query Image")
        cv2.moveWindow("Query Image",400,50)
        #cv2.resizeWindow("Query Image", 400, 50)
        cv2.imshow("Query Image",query_img)

        #res=[]
        #for n in range(300):
        #    ltrb=200+np.random.randint(300,size=4)
        #    ltrb[[0,1]]+=500
        #    ltrb[[2, 3]] += ltrb[[0, 1]]
        #    l,t,r,b=ltrb.tolist()
        #    res.append(img[t:b,l:r,:])
        #    print res[-1].shape
        res=db.qbe(query_img)
        plot_results(res,50,(1000,500))
        cv2.waitKey(10)

    def qbs():
        res = db.qbe(query_img)
        print("QBS")
        # grab a reference to the image panels
        global panelA, panelB

        # open a file chooser dialog and allow the user to select an input
        # image
        tkSimpleDialog.askstring("Query Entry", "Which word are you searching for")


    win_root = Tkinter.Tk()
    panelA = None
    panelB = None

    # create a button, then when pressed, will trigger a file chooser
    # dialog and allow the user to select an input image; then add the
    # button the GUI
    btn_qbe = Tkinter.Button(win_root, text="Query By Example", command=qbe)
    btn_qbe.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")
    btn_qbs = Tkinter.Button(win_root, text="Query By String", command=qbs)
    btn_qbs.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")

    # kick off the GUI
    win_root.mainloop()
