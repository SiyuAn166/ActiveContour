import numpy as np


def line_energy(image):
    # implement line energy (i.e. image intensity)
    return image
    
def edge_energy(image):
    # implement edge energy (i.e. gradient magnitude)
    dy,dx = np.gradient(image)
    return -np.sqrt(dy**2 + dx**2)
    
    
def term_energy(image):
    #implement term energy (i.e. curvature)
    Cy,Cx = np.gradient(image)
    Cyy, Cyx = np.gradient(Cy)
    Cxy, Cxx = np.gradient(Cx)
    return (Cyy * Cx ** 2 - 2 * Cxy * Cx * Cy + Cxx * Cy ** 2) / ((1e-4 + Cx ** 2 + Cy ** 2)**1.5)
    
def external_energy(image, w_line, w_edge, w_term):
    #implement external energy
    return w_line*line_energy(image) + w_edge*edge_energy(image) + w_term*term_energy(image)

