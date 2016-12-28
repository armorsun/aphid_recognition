import imutils

def pyramid(image,scale=1.5, min_size=(200,200)):
    yield image

    while True:
        w=int(image.shape[1]/scale)
        image=imutils.resize(image,width=w)
        if image.shape[0]<min_size[1] or image.shape[1]<min_size[0]:
            break

        yield image

def sliding_window(image, step_size, window_size):
    for y in xrange(0, image.shape[0], step_size):
        for x in xrange(0, image.shape[1],step_size):
            yield(x,y,image[y:y+window_size[1],x:x+window_size[0]])


