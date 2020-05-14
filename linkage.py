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
              draw_tower = False, 
              view_box = 15, 
              show = True):
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
            
        
        x_points.append(p0[0])
        x_points.append(p1[0])
        
        y_points.append(p0[1])
        y_points.append(p1[1])
    
    p3 = get_tri(p1, p0, 
                 l0 = np.mean(lengths[0][0] + lengths[0][1]), 
                 l1 = np.mean(lengths[0][0] + lengths[0][1]))
    
    if type(p3) != type(None):
        y_points.append(p3[1])
        x_points.append(p3[0])
        y_points.append(p3[1])
        x_points.append(p3[0])
        
        x = np.asarray(x_points)
        y = np.asarray(y_points)
    else:
        return None
    
    if draw_tower:
        c1 = [(False, True)[(i%4 == 3) ^ (i%4 == 1)] for i in range(len(x_points))]
        c2 = [(False, True)[(i%4 == 2) ^ (i%4 == 0)] for i in range(len(x_points))]
        
        plt.ylim(top = view_box, bottom = -view_box/2)
        plt.xlim(left = -view_box, right = view_box)
        plt.scatter(x_points, y_points, s = 1)
        plt.scatter([A0[0], A1[0]], [A0[1], A1[1]], s = 1)
        plt.plot(x[c2], y[c2])
        plt.plot(x[c1], y[c1])
        
        if show:
            plt.show()
        
    return [p3, [x, y]]

def get_trace(A0, lengths, num_samples, draw_tower = False):
    """
    Takes the anchorpoint at [0, 0]. The other anchor point, A1, passes through 
    all values in the range [0, [0, (lengths[0][0] + lengths[0][1])/2]]. 
    
    We store the values that the top of the tower passes through
    """
    
    max_sep = (lengths[0][0] + lengths[0][1])
    min_sep = (lengths[0][0] + lengths[0][1])/5
    
    points = []
    
    for i, d in enumerate(np.linspace(max_sep, min_sep, num = num_samples)):
        if draw_tower:
            tower_points = get_tower(A0, [A0[0] + d, A0[1]], lengths, 
                                     draw_tower = i%(100) == 0)
        else:
            tower_points = get_tower(A0, [A0[0] + d, A0[1]], lengths, 
                                     draw_tower = False)
        
        if type(tower_points) == type(None):
            break
        else:
            points.append(tower_points[0])
            
        """
        If there is a large jump in the trace, then something non-physical 
        has occurred, and so we break
        """
        if (i > 10) and (abs(np.linalg.norm(points[i] - points[i-1])) > 1):
                break
            
    points = np.asarray(points)
    return points

def disp_tower(lengths,
               A0 = None, 
               A1 = None,
               view_box = 15, 
               show = False):
    
    if (A0 == None) and (A1 == None):
            """
            If no anchor points are given, use the default
            (have to set the default here, because the default depends
             on a given variable)
            """
            A0 = np.asarray([0, 0])
            A1 = np.asarray([(lengths[0][0] + lengths[0][1])/2, 0])
            
    get_tower(A0, A1, lengths, 
              draw_tower = True, view_box = view_box, show = show)
    
    
def disp_trace(trace,
               label = '',
               cmap = 'winter'):
    
    plt.scatter(trace[:, 0], 
                trace[:, 1], 
                c = np.linspace(0, 1, num = len(trace)),
                label = label,
                cmap = cmap, 
                s = 2)
    if label != '':
        plt.colorbar(label = label)
    #plt.show()
    
def disp_traces(traces, 
                label = '',
                cmap = 'winter'):
    
    for trace in traces:
        disp_trace(trace, 
                   label = '',
                   cmap = 'winter')


def make_heart(num_samples):
    """
    Makes a heart shaped trace
    """
    thetas = [x for x in np.linspace(3*np.pi/2, -np.pi/2, num_samples)]
    
    def heart(x):
        return 2 - 2*np.sin(x) + np.sin(x)*np.sqrt(np.abs(np.cos(x)))/(np.sin(x) + 1.5)
    
    thetas = [[np.cos(x)*heart(x),np.sin(x)*heart(x)] for x in thetas]
    return np.asarray(thetas)

def make_dna(num_samples):
    """
    Makes a veritcal dna strand
    
    Basically just two vertical sinusoidal waves offset by 180 degrees. 
    Then we got some ladder rungs connecting the two twisting edges. 
    """
    
    y_vals = np.linspace(0, 1.5*np.pi, num_samples)
    vert_cos = np.asarray([[np.cos(y), y] for y in y_vals])
    vert_sin = np.asarray([[np.sin(y), y] for y in y_vals])
    
    dna = [vert_cos, vert_sin]
    
    for d in np.linspace(0.25, 1.5*np.pi-0.25, 10):
        rung_width = abs(np.cos(d) - np.sin(d))
        
        if rung_width > 0.1:
            temp_rung = np.linspace(np.cos(d), np.sin(d), np.floor(rung_width/(1.5*np.pi/num_samples)))
            dna.append(np.asarray([[r, d] for r in temp_rung]))
    
    return dna
    
    
def constrain(x, interval):
    """
    Always gonna need a constrain function!
    """
    if (x <= interval[0]):
        return interval[0]
    if (x >= interval[1]):
        return interval[1]
    return x

def ho_reflection(x, max_x):
    """
    If x is less than max_x, return x.
    Otherwise, return 2*max_x - x
    """
    if x >= max_x:
        return 2*max_x - x
    
    return x
    
def vert_reflect_tower(lengths):
    """
    For each quad in the tower, we perform the permutation
    [l1, l2, l3, l4] --> [l2, l1, l4, l3]
    """
    new_lengths = []
    
    for length in lengths:
        new_lengths.append([length[1], 
                            length[0], 
                            length[3], 
                            length[2]])
    return new_lengths

def disp_moving_towers(towers, 
                       pos_offsets = None,
                       time_offsets = None, 
                       num_samples = 100, 
                       frame_rate = 1):
    
    """
    Takes a list of towers and a list of positional offsets, 
    and draws each tower, shifted by the positional offset, as it articulates
    its trace. 
    
    the time_offsets variable is a list of integers [t0, t1, ...] representing the 
    frame tn at which the nth tower should start drawing. 
    
    the frame_rate is the number of frames to compute before displaying something. 
    So the number of frames will be something like num_samples/frame_rate
    """
    
    if time_offsets == None:
        time_offsets = np.zeros(len(towers))
        
    if pos_offsets == None:
        pos_offsets = np.zeros((len(towers), 2))
    
    traces = [[[None, None]] for x in range(len(towers))]
    
    num_frames = num_samples + int(max(time_offsets)) + 2
    
    for frame in range(num_frames):
        
        for i in range(len(towers)):
            tow_n = towers[i]
            p_n = pos_offsets[i]
            t_n = time_offsets[i]
                
            if (t_n <= frame) and ((frame % frame_rate) == 0):
                d = constrain((frame - t_n)/num_samples, [0,1])
  
                max_sep = (tow_n[0][0] + tow_n[0][1])
                min_sep = (tow_n[0][0] + tow_n[0][1])/2
                
                d_new = (1 - d)*max_sep + (d)*min_sep

                
                tower = get_tower(A0 = np.asarray(p_n),
                                  A1 = np.asarray([p_n[0] + d_new, p_n[1]]), 
                                  lengths = tow_n, 
                                  draw_tower = False, 
                                  show = False)
                ax = plt.gca()
                ax.axes.xaxis.set_visible(False)
                ax.axes.yaxis.set_visible(False)
                
                plt.ylim(top = 12, bottom = -1)
                plt.xlim(left = -2, right = 6)
                
                if (type(tower) != type(None)) and (d_new > min_sep):
                    x, y = tower[1]
                    
                    traces[i].append(tower[0])
                    c1 = [(False, True)[(i%4 == 3) ^ (i%4 == 1)] for i in range(len(x))]
                    c2 = [(False, True)[(i%4 == 2) ^ (i%4 == 0)] for i in range(len(x))]
                    
                    plt.scatter(x, y, s = 1)
                    plt.plot(x[c2], y[c2])
                    plt.plot(x[c1], y[c1])
                
                plt.scatter(np.asarray(traces[i])[:, 0], 
                            np.asarray(traces[i])[:, 1], 
                            c = np.linspace(0, 1, num = len(traces[i])),
                            cmap = 'rainbow_r', 
                            s = 2)
        plt.show()
                

disp_traces(make_dna(100))
best_lh_tower = np.loadtxt("best_lh_tower.csv")
best_rh_tower = np.loadtxt("best_rh_tower.csv")

test_tower = [[1, 1, 1, 1], 
              [1, 1, 1, 1], 
              [1, 1, 1, 1], 
              [1, 1, 1, 1], 
              [1, 1, 1, 1]]

towers = [best_lh_tower, best_rh_tower]

disp_moving_towers(towers, 
                    pos_offsets = [[0, 0], [.9, -.5]], 
                    time_offsets = [30, 50], 
                    num_samples = 180, 
                    frame_rate = 1)





    