import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib import animation
import intervals as I
import os
import re 

from pylab import rcParams
from progress_bar import ProgressBar


plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
rcParams['figure.figsize'] = 15, 15

"""
A set of functions that allow us to instantiate linkages, as well as 
draw the traces for those linkages. 
"""

def get_tri(p0, p1, l0, l1):
    '''
    a basic way to solve for the final coordinate of a triangle defined by SSS
    
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
    
    p3 = get_tri((p1, p0)[len(lengths)%2 == 0], (p0, p1)[len(lengths)%2 == 0], 
                 l0 = (lengths[-1][3] + lengths[-1][2])/2, 
                 l1 = (lengths[-1][2] + lengths[-1][3])/2)
    
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


"""
Rather than computing a trace for all t values in [max_sep, min_sep], we are going 
to narrow this interval down to hopefully decrease computation time
"""

def get_t_final_interval(lengths):
    """
    Returns the interval on which the trace will be well defined (the tower will be unbroken)
    Where T_final = intersection_{j=1}^{n} [|t_{j,2} -t_{j, 1}|, t_{j,2} + t_{j, 1}]
    """
    def get_base_interval(quad):
        """
        Returns range of motion for the bottom of the quad
        """
        return [abs(quad[1] - quad[0]), quad[1] + quad[0]]
    
    def get_top_interval(quad):
        """
        Returns the range of motion for the top of the quad
        """
        return [abs(quad[2] - quad[3]), quad[2] + quad[3]]
    
    def get_t_prime(quad, t):
        """
        Returns the distance between the top two endpoints as a function of the quad and t 
        """
        l1, l2, l3, l4 = quad
        a = l3**2 + l4**2
        b = l4*l3*(l1**2 + l2**2 - t**2)
        c = l1*l2
        
        return np.sqrt(a - b/c)
    
    def intersection(A, B):
        """
        Returns the intersection of A and B (subsets of the real numbers)
        """
        AcapB = I.closed(A[0], A[1]) & I.closed(B[0], B[1])
        
        if AcapB.lower == I._PInf():
            return None #here None indicates that the intersection is empty
        return [AcapB.lower, AcapB.upper]
        
    
    #getting top interval
    top_interval = get_base_interval(lengths[0])
    for i, quad in enumerate(lengths[1:-1]):
        next_interval = [get_t_prime(quad, top_interval[0]), 
                         get_t_prime(quad, top_interval[1])]
        top_interval = intersection(next_interval, 
                                    get_base_interval(lengths[i+1]))
     
    #now taking the preimage of the top interval
    ho_refl_tower = ho_reflect_tower(lengths)
    base_interval = top_interval
    for i, quad in enumerate(ho_refl_tower[1:-1]):
        next_interval = [get_t_prime(quad, top_interval[0]), 
                         get_t_prime(quad, top_interval[1])]
        base_interval = intersection(next_interval, 
                                     get_base_interval(lengths[i+1]))
        
    return base_interval


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
            A1 = np.asarray([(lengths[0][0] + lengths[0][1])/3, 0])
            
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

def make_dna(num_samples, 
             rungs = 5, 
             dpos = [0, 0], 
             dscale = [1, 1]):
    """
    Makes a veritcal dna strand
    
    Basically just two vertical sinusoidal waves offset by pi/2. 
    Then we got some ladder rungs connecting the two twisting edges. 
    """
    
    y_vals = np.linspace(0, 1.5*np.pi, num_samples)
    vert_cos = np.asarray([[np.cos(y)*dscale[0]+dpos[0], y*dscale[1]+dpos[1]] for y in y_vals])
    vert_sin = np.asarray([[np.sin(y)*dscale[0]+dpos[0], y*dscale[1]+dpos[1]] for y in y_vals])
    
    dna = [vert_cos, vert_sin]
    
    for d in np.linspace(0.25, 1.5*np.pi-0.25, rungs):
        rung_width = abs(np.cos(d) - np.sin(d))
        
        if rung_width > 0.1:
            temp_rung = np.linspace(np.cos(d), np.sin(d), int(rung_width/(1.5*np.pi/num_samples)))
            dna.append(np.asarray([[r*dscale[0]+dpos[0], d*dscale[1]+dpos[1]] for r in temp_rung]))
    
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
    new_lengths = [[l[1],l[0],l[3],l[2]] for l in lengths]

    return new_lengths

def ho_reflect_tower(lengths):
    """
    make the top of the tower be the new bottom of the tower
    """
    new_lengths = lengths[-1::-1]
    new_lengths = [[l[3], l[2], l[1], l[0]] for l in new_lengths]
    return new_lengths

def scale_tower(lengths, scalar):
    #just globally scales the tower
    return (np.asarray(lengths)*scalar).tolist()

def adjust_pencil(lengths, scalar):
    #scales the top two lengths of the tower (absusive notation, abuse of)
    #scaling these two lengths changes how the curve is traced, 
    temp_lengths = lengths[:-1]
    temp_lengths.append([lengths[-1][0], 
                         lengths[-1][1], 
                         lengths[-1][2]*scalar, 
                         lengths[-1][3]*scalar])
    return temp_lengths
    
def disp_moving_towers(towers, 
                       pos_offsets = None,
                       time_offsets = None, 
                       num_samples = 100, 
                       frame_rate = 1, 
                       save_folder = "", 
                       background_image = "",
                       bounds = [[-3, 7], [11, -1]]):
    
    """
    Takes a list of towers and a list of positional offsets, 
    and draws each tower, shifted by the positional offset, as it articulates
    its trace. 
    
    the time_offsets variable is a list of integers [t0, t1, ...] representing the 
    frame tn at which the nth tower should start drawing. 
    
    the frame_rate is the number of frames to compute before displaying something. 
    So the number of frames will be something like num_samples/frame_rate
    
    bounds is for setting xlim ylim with [[left, right], [top, bottom]]
    """
    

    if time_offsets == None:
        time_offsets = np.zeros(len(towers))
        
    if pos_offsets == None:
        pos_offsets = np.zeros((len(towers), 2))
    
    traces = [[[0, 0]] for x in range(len(towers))]
    
    num_frames = num_samples + int(max(time_offsets)) + 2
    
    im = None
    xscale, yscale = 1.0, 1.0
    
    if save_folder != "":
        try:
            os.mkdir(save_folder)
        except FileExistsError:
            pass
    
    if background_image != "":
        im = Image.open(background_image).rotate(180)
        
        xscale = im.width/abs(bounds[0][0] - bounds[0][1])
        yscale = im.height/abs(bounds[1][0] - bounds[1][1])
    
    print("Drawing moving towers...\n")
    progress = ProgressBar(num_frames, fmt = ProgressBar.FULL)
    for frame in range(num_frames):
        progress.current += 1
        progress.__call__()
        
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        
        plt.axis('off')
        
        if im == None:
            plt.xlim(left = bounds[0][0], right = bounds[0][1])
            plt.ylim(top = bounds[1][0], bottom = bounds[1][1])
        else:
            plt.xlim(left = 0, right = im.width)
            plt.ylim(top = im.height, bottom = 0)
            
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
                
                if background_image != "":
                    plt.imshow(im, origin = 'lower')
                
                if (type(tower) != type(None)) and (d_new > min_sep):
                    x, y = tower[1]
                    
                    traces[i].append(tower[0])
                    c1 = [(False, True)[(i%4 == 3) ^ (i%4 == 1)] for i in range(len(x))]
                    c2 = [(False, True)[(i%4 == 2) ^ (i%4 == 0)] for i in range(len(x))]
                    
                    plt.scatter(x, y, s = 1)
                    plt.plot(np.asarray(x[c2])*xscale, 
                             np.asarray(y[c2])*yscale, 
                             linewidth = 3.5, 
                             c = ('w', 'r')[im == None])
                    
                    plt.plot(np.asarray(x[c1])*xscale, 
                             np.asarray(y[c1])*yscale, 
                             linewidth = 3.5, 
                             c = ('w', 'b')[im == None])
                    
                plt.scatter(np.asarray(traces[i])[:, 0]*xscale, 
                            np.asarray(traces[i])[:, 1]*yscale, 
                            c = np.linspace(0, 1, num = len(traces[i])),
                            cmap =  ('spring', 'rainbow_r')[im == None], 
                            s = 20)
            
            
        if (save_folder != "") and ((frame % frame_rate) == 0):
            plt.savefig("{}/frame_{}.jpg".format(save_folder, frame),
                        bbox_inches ='tight', 
                        dpi = 400)
            
        if ((frame % frame_rate) == 0):
            plt.show()
            
                    
def make_movie(dir_of_ims, target_name):

    files_in_dir = os.listdir(dir_of_ims)
    
    """
    os.listdir gives you a list of all the file names in the provided dir, but the order
    of the file names in the list isn't the same as the way the files appear 
    in the directory. So we gotta use the strength of pandas.sort_values to help
    """
    images = pd.DataFrame({'fnames':files_in_dir, 
                           'fnumbers':[int(re.findall("\d+", f)[0]) for f in files_in_dir]})
    
    images = images.sort_values(by=['fnumbers'])
    sorted_names = list(images['fnames'].values)
    
    #making the trace hang on longer in the end
    last_im_name = sorted_names[-1]
    for _ in range(60):
        sorted_names.append(last_im_name)
        
    """
    matplotlib can be annoying and i'm just trying to make it so that the
    white padding around the image is gone -- it might be overkil and i just
    tried things until it worked
    """
    fig = plt.figure(frameon = False)
    fig.set_size_inches(15, 15)
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
            
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)
    
    images = [] 
    progress = ProgressBar(len(sorted_names), fmt = ProgressBar.FULL)
    print("Making a movie, compiling images...")
    for name in sorted_names:
        progress.current += 1
        progress.__call__()
        
        temp_im = Image.open("{}/{}".format(dir_of_ims, name)).rotate(180)
        plt.xlim(left = 0, right = temp_im.width)
        plt.ylim(top = temp_im.height, bottom = 0)

        plt.axis('off')
        
        plt_im = ax.imshow(temp_im, aspect = temp_im.width/temp_im.height)
        images.append([plt_im])
        
    print("\nWriting and saving animation...(this may take a while)")
    ani = animation.ArtistAnimation(fig, images, interval = 50)
    writer = animation.FFMpegWriter(fps = 30)
    ani.save(target_name, writer = writer)
    print("Done!")
    

#example demo of the linkages tracing out a heart

# img_loc = "/home/oliver/Documents/linkage_ga/cute_images/goats.jpeg"
# test_tower = [[1, 1.2, 1, 1.2], [1, 1.3, 1.2, 1.2], [1, 1.3, 1.2, 1.2], [1, 1, 1, 1]]


# disp_moving_towers([adjust_pencil(test_tower, 2), 
#                     scale_tower(test_tower, 1.15), 
#                     test_tower])

# best_lh_tower = np.loadtxt("best_lh_tower.csv")
# best_rh_tower = np.loadtxt("best_rh_tower.csv")

# towers = [best_lh_tower, best_rh_tower]

# disp_moving_towers(towers, 
#                     pos_offsets = [[3, 1], [3.9, .5]], 
#                     time_offsets = [0, 0], 
#                      num_samples = 500, 
#                     save_folder = "heart_output", 
#                     background_image = img_loc)


#make_movie('heart_output', 'heart.mp4')


#example demo of the linkages drawing out dna

# dna_towers_names = os.listdir("dna_output/")
# dna_towers = []

# for name in dna_towers_names:
#     dna_towers.append(np.loadtxt("dna_output/{}".format(name)))
    
# pos_offsets = [[3, 1] for _ in dna_towers]
# disp_moving_towers(dna_towers, 
#                    pos_offsets = pos_offsets,
#                    num_samples = 200)



    