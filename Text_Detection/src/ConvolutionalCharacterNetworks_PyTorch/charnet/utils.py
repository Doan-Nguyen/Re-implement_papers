from imports import * 


def tensor_to_numpy(*tensor):
    """
    This function to convert data from torch tensor to numpy array.

    Parameters:
        - tensor:
    Return:
        - A numpy array
    """
    results = []
    for t in tensor:
        if t is None:
            results.append(None)
        else:
            results.append(t.cpu().numpy())
    return results

# def rotate_rect(x1, y1, x2, y2, degree, center_x, center_y):

    
