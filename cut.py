import random
from datetime import datetime
import Image
im=Image.open("/home/sunny/Documents/lab/leaf/leaf_keras/6.jpg")

w,h=im.size
#print(w,h)

random.seed(datetime.now())
w=w-200
h=h-200
#print(w,h)

for x in xrange(1,1001):
    a=random.randint(0,w)
    b=random.randint(0,h)
    #print(a,b)
    newim=im.crop((a,b,a+200,b+200))
    newim=newim.resize((128,128))
    filename='im_6/im_128_1000_3/c_'+str(x)+'.jpg'
    newim.save(filename,"JPEG")



