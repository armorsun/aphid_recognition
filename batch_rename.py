import os
path=os.getcwd()+'/trainingData/test/aphids/'
print(path)
files=os.listdir(path)
newpath=os.getcwd()+'/trainingData/test/aphids_r/'
n=1
for filename in files:
	oldpath=os.path.join(path,filename)
    	newname=os.path.join(newpath, 'aphids'+ str(n)+'.jpg')
	#print(newname)	
	#print(oldpath)
    	os.rename(oldpath,newname)		
	#print(n)
    	n=n+1


