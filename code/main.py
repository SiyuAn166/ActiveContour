import cv2
import numpy as np
import matplotlib.pyplot as plt
from external_energy import external_energy
from internal_energy_matrix import get_matrix




def click_event(event, x, y, flags, params):
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        #save point
        xs.append(x)
        ys.append(y)

        #display point
        cv2.circle(img, (x, y), 3, 128, -1)
        cv2.imshow('image', img)


if __name__ == '__main__':
    #point initialization
    img_path = '../images/star.png'
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h,w = img.shape
    xs = []
    ys = []
    image_blur = cv2.GaussianBlur(img.copy(),(5,5),1,1)
    
    
    cv2.imshow('image', img)
    cv2.setMouseCallback('image', click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    assert len(xs) > 1, "Invalid inputs."
    
    xs.append(xs[0]), ys.append(ys[0]) # implement part 1: interpolate between the selected points
    inter_size = 100 # interpolation size
    x_ext,y_ext = [],[]
    for i in range(1, len(xs)):
        x_b,y_b = [],[]
        x_b.extend(np.linspace(xs[i-1],xs[i],inter_size,endpoint=False))
        y_b.extend(np.linspace(ys[i-1],ys[i],inter_size,endpoint=False))
        x_ext.append(xs[i-1]), y_ext.append(ys[i-1])
        x_ext.extend(x_b), y_ext.extend(y_b)
        x_ext.append(xs[i]), y_ext.append(ys[i])
    xs, ys = x_ext, y_ext
    xs, ys = np.array(xs).astype(int), np.array(ys).astype(int)
    
    alpha = 4e-2 # dist
    beta = 1     # stiffness
    gamma = 1
    # step
    kappa = 1e-1
    
    w_line = 5e-2
    w_edge = 5e-1
    w_term = 5e-1 #curvature
    n_iteration = 8000
    
    
    x_init, y_init = xs.copy(), ys.copy()
    #get matrix
    num_points = len(xs)
    M = get_matrix(alpha, beta, gamma, num_points)
    #get external energy
    E = external_energy(image_blur, w_line, w_edge, w_term)
    fy, fx = np.gradient(E)
    x,y = xs.copy(), ys.copy()
    
    
    animation = True
    # animation = False
    
    if animation:
        #animation
        fig = plt.figure()
        plt.ion()
        for k in range(n_iteration):
            i,j = y.astype(int),x.astype(int)
            x = M @ (gamma * x - kappa * fx[i,j])
            y = M @ (gamma * y - kappa * fy[i,j])
            x[x<0]=0
            x[x>w-1] = w-1
            y[y<0] = 0
            y[y>h-1]=h-1
            
            if k % 100 == 0:
                ax  = fig.add_subplot(111)
                ax.imshow(img, cmap=plt.cm.gray)
                ax.plot(x,y,'.r',lw=1)
                plt.pause(0.1)
                ax.remove()
            
        xs,ys=x,y
        plt.ioff()
    
    else:
        # static image
        for k in range(n_iteration):
            i,j = y.astype(int),x.astype(int)
            x = M @ (gamma * x - kappa * fx[i,j])
            y = M @ (gamma * y - kappa * fy[i,j])
            x[x<0]=0
            x[x>w-1] = w-1
            y[y<0] = 0
            y[y>h-1]=h-1
        xs,ys=x,y
        
        fig = plt.figure()
        ax  = fig.add_subplot(111)
        ax.imshow(img, cmap=plt.cm.gray)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0,img.shape[1])
        ax.set_ylim(img.shape[0],0)
        ax.plot(x_init,y_init,'-b',lw=1)
        ax.plot(xs,ys,'.r',lw=3)
        plt.show()
        
    
    
    