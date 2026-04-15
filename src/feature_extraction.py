import numpy as np
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import shannon_entropy

def extract_glcm_features(image):
    """Extract GLCM texture features as described in Paper 1"""
    img_uint8 = (image * 255).astype(np.uint8)
    glcm = graycomatrix(img_uint8, distances=[1],
                        angles=[0], levels=256,
                        symmetric=True, normed=True)
    features = {
        'energy':       graycoprops(glcm, 'energy')[0, 0],
        'contrast':     graycoprops(glcm, 'contrast')[0, 0],
        'correlation':  graycoprops(glcm, 'correlation')[0, 0],
        'homogeneity':  graycoprops(glcm, 'homogeneity')[0, 0],
        'dissimilarity':graycoprops(glcm, 'dissimilarity')[0, 0],
    }
    return features

def extract_statistical_features(image):
    """Statistical features from Paper 4"""
    flat = image.flatten()
    return {
        'mean':    np.mean(flat),
        'std':     np.std(flat),
        'entropy': shannon_entropy(image),
        'skewness':float(np.mean(((flat - np.mean(flat)) / (np.std(flat) + 1e-6))**3)),
        'kurtosis':float(np.mean(((flat - np.mean(flat)) / (np.std(flat) + 1e-6))**4)),
    }

def get_feature_vector(image):
    """Combine all features into one vector"""
    glcm = extract_glcm_features(image)
    stat = extract_statistical_features(image)
    combined = {**glcm, **stat}
    return np.array(list(combined.values()))