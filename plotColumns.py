import matplotlib.pyplot as plt
import numpy as np

def plotColumns(a, b, labelA, labelB, title=None):
    '''
    incoming parameters are numpy arrays
    '''
    length = a.shape[0]+1
    x = np.arange(1, length, 1)
    # The two series could have different scales so need separate axes
    low  = a.min()*0.9
    high = a.max()*1.1
    y1 = np.linspace(start=low, stop=high, num=length)
    
    low  = b.min()*0.9
    high = b.max()*1.1
    y2 = np.linspace(start=low, stop=high, num=length)
    
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(x, a, 'r-')
    ax2.plot(x, b, 'b-')
    ax1.set_xlabel('')
    ax1.set_ylabel(labelA, color='r')
    ax2.set_ylabel(labelB, color='b')
    if title:
        plt.title(title)
    plt.show()