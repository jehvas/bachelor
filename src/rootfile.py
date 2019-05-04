import os

ROOTPATH = os.path.dirname(__file__) + '/../'

if os.path.isdir(ROOTPATH + 'out/'):
    print('Running locally. Using standard ROOTPATH')
else:
    if ROOTPATH.find('src') != -1:
        ROOTPATH = os.path.dirname(__file__) + '/../../../drive/My Drive/Bachelor/'
    else:
        ROOTPATH = os.path.dirname(__file__) + '/../../drive/My Drive/Bachelor/'
    print('Running on COLAB. Using alertnate ROOTPATH')
