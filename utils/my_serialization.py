import pickle


def loadpkl(path):
    with open(path,'rb') as f:
        return pickle.load(f)
def savepkl(path,data):
    with open(path,'wb') as f:
        return pickle.dump(data,f,2)