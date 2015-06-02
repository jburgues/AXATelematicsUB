import numpy as np

#%%
def dsquared_line_points(P1, P2, points):
    '''
    Calculate only squared distance, only needed for comparison
    http://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
    '''
    xdiff = P2[0] - P1[0]
    ydiff = P2[1] - P1[1]
    nom  = (
        ydiff*points[:,0] - \
        xdiff*points[:,1] + \
        P2[0]*P1[1] - \
        P2[1]*P1[0]
    )**2
    denom = ydiff**2 + xdiff**2
    return np.divide(nom, denom)

def rdp_numpy(M, epsilon = 0):

    # initiate mask array
    # same amount of points
    mask = np.empty(M.shape[0], dtype = bool)

    # Assume all points are valid and falsify those which are found
    mask.fill(True)

    # The stack to select start and end index
    stack = [(0 , M.shape[0]-1)]

    while (len(stack) > 0):
        # Pop the last item
        (start, end) = stack.pop()

        # nothing to calculate if no points in between
        if end - start <= 1:
            continue

        # Calculate distance to points
        P1 = M[start]
        P2 = M[end]
        points = M[start + 1:end]
        dsq = dsquared_line_points(P1, P2, points)

        mask_eps = dsq > epsilon**2

        if mask_eps.any():
            # max point outside eps
            # Include index that was sliced out
            # Also include the start index to get absolute index
            # And not relative 
            mid = np.argmax(dsq) + 1 + start
            stack.append((start, mid))
            stack.append((mid, end))

        else:
            # Points in between are redundant
            mask[start + 1:end] = False

    return mask 
