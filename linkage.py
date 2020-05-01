import numpy as np
import matplotlib.pyplot as plt

"""
A set of functions that allow us to instantiate linkages, as well as 
draw the traces for those linkages. 
"""

def get_tri(p0, p1, l0, l1):
    '''
    p1, p2 are numpy arrays representing the 2D location of the
    points of the linkage. 
    
    l1, l2 are the side lengths in question
    
    Returns the left-handed solution to the triangle problem, 
    if the solution doesn't exist, then it will return none.
    '''
    
    d = np.linalg.norm(p0 - p1)
    
    if np.abs(d) < 1e-18:
        return None
    
    if d > l0 + l1:
        #where no solution exists
        return None
    
    a = ((l0**2) - (l1**2) + (d**2))/(2*d)
    h = np.sqrt(abs(l0**2 - a**2))
    
    q = (p1 - p0) * (a/d)
    p2 = p0 + q
    
    x = p2[0] - h * (p1[1] - p0[1])/d
    y = p2[1] - h * (p2[0] - p1[0])/(d/2)
    
    return np.asarray([x, y])

def get_quad(p0, p1, L):
    """
    p0, p1 are the anchor points
    
    L is a list of four lengths. 
    
    This basically just makes the scissor shaped thing
    """
    l0, l1, l2, l3 = L
    pC = get_tri(p0, p1, l0, l1)
    
    if type(pC) == type(None):
        #where no solution exists, don't continue
        return None
    
    norm_p1_pC = (pC-p1) / np.linalg.norm(p1 - pC)
    norm_p0_pC = (pC-p0) / np.linalg.norm(p0 - pC)
    
    p2 = (norm_p0_pC*(l0+l2)) + p0
    p3 = (norm_p1_pC*(l1+l3)) + p1
    
    return [p2, p3]
    
def get_tower(A0, A1, lengths, 
              draw_tower = False, view_box = 15):
    """
    A0, A1 are the anchor points
    lengths is a list of lengths of size (n, 4)
    
    Stacks quads into a tower
    """
    p0, p1 = None, None
    
    try:
        p0, p1 = get_quad(A0, A1, lengths[0])
    except:
        return None
    
    x_points = []
    y_points = []
    
    if draw_tower:
        x_points.append(A0[0])
        x_points.append(A1[0])
        x_points.append(p0[0])
        x_points.append(p1[0])
        
        y_points.append(A0[1])
        y_points.append(A1[1])
        y_points.append(p0[1])
        y_points.append(p1[1])
        
    for i, L in enumerate(lengths[1:]):
        if type(p0) == type(None):
            return None
        
        """
        Depending on the parity of the index of the quad
        we are trying to append to the top of the stack, 
        we have to flip the order of the endpoints to ensure
        that the tower grows in the way we would expect it to
        """
        if i%2 == 0:
            try:
                p1, p0 = get_quad(p1, p0, L)
            except:
                return None
        if i%2 == 1:
            try:
                p0, p1 = get_quad(p0, p1, L)
            except:
                return None
            
        if draw_tower:
            x_points.append(p0[0])
            x_points.append(p1[0])
            
            y_points.append(p0[1])
            y_points.append(p1[1])
            
    if draw_tower:
        c1 = [(False, True)[(i%4 == 3) ^ (i%4 == 1)] for i in range(len(x_points))]
        c2 = [(False, True)[(i%4 == 2) ^ (i%4 == 0)] for i in range(len(x_points))]
        
        x = np.asarray(x_points)
        y = np.asarray(y_points)
        
        plt.ylim(top = view_box, bottom = -view_box/2)
        plt.xlim(left = -view_box, right = view_box)
        plt.scatter(x_points, y_points, s = 1)
        plt.scatter([A0[0], A1[0]], [A0[1], A1[1]], s = 1)
        plt.plot(x[c2], y[c2])
        plt.plot(x[c1], y[c1])
        plt.show()
        
    return [p0, p1]


def get_trace(A0, A1, lengths, num_samples, draw_tower = False):
    """
    Takes the anchorpoint at [0, 0]. The other anchor point, A1, passes through 
    all values in the range [0, [0, (lengths[0][0] + lengths[0][1])/2]]. 
    
    We store the values that the top of the tower passes through
    """
    
    max_sep = (lengths[0][0] + lengths[0][1])
    min_sep = (lengths[0][0] + lengths[0][1])/4
    
    points = []
    
    for i, d in enumerate(np.linspace(max_sep, min_sep, num = num_samples)):
        if draw_tower:
            tower_points = get_tower(A0, [A0[0] + d, A0[1]], lengths, 
                                     draw_tower = i%(100) == 0)
        if not(draw_tower):
            tower_points = get_tower(A0, [A0[0] + d, A0[1]], lengths, 
                                     draw_tower = False)
        
        if type(tower_points) != type(None):
            points.append(tower_points[1])
            
            
    points = np.asarray(points)
    return points

def disp_tower(lengths, view_box = 15):
    """
    A roundabout way of drawing a simple tower:
    """
    A0 = np.asarray([0, 0])
    A1 = np.asarray([(lengths[0][0] + lengths[0][1])/2, 0])
    
    get_tower(A0, A1, lengths, draw_tower = True, view_box = view_box)
    
    
def disp_trace(trace,
               label = '',
               cmap = 'winter', 
               view_box = 15):
    
    plt.scatter(trace[:, 0], 
                trace[:, 1], 
                c = np.linspace(0, 1, num = len(trace)),
                label = label,
                cmap = cmap, 
                s = 2)
    plt.colorbar(label = label)
    #plt.show()


def make_heart(num_samples):
    """
    Makes a heart shaped trace
    """
    thetas = [x for x in np.linspace(3*np.pi/2, -np.pi/2, num_samples)]
    
    def heart(x):
        return 2 - 2*np.sin(x) + np.sin(x)*np.sqrt(np.abs(np.cos(x)))/(np.sin(x) + 1.5)
    
    thetas = [[np.cos(x)*heart(x),np.sin(x)*heart(x)] for x in thetas]
    return np.asarray(thetas)


"""
'''
Hardcoding a tower and displaying the trace for that tower
'''
p0 = np.asarray([0, 0])
p1 = np.asarray([1, 0])
rtt = np.sqrt(2)

L = [[rtt, rtt, rtt, rtt],
     [rtt, rtt, rtt, 1.2],
     [rtt, rtt, rtt, 1.2],
     [rtt, rtt, rtt, 1.2], 
     [rtt, rtt, 1.2, 1.2],
     [rtt, rtt, 1.2, rtt],
     [rtt, rtt, 1.2, rtt], 
     [rtt, rtt, rtt, rtt]]

points = []
c = []

for i, d in enumerate(np.linspace(5, 0, num = 5000)):
    new_p1 = np.asarray([p1[0]+d, p1[1]])
    if i % 50 == 0:
        get_tower(p0, new_p1, L, draw_tower = True)
    new_points = get_tower(p0, new_p1, L)
    
    if type(new_points) != type(None):
        points.append(new_points[1])
        c.append(d)
    
points = np.asarray(points)

plt.scatter(points[:, 1], points[:, 0], c = c)
plt.colorbar()
plt.show()
"""




    