import pickle

DEFAULT_LUT = 'data/mapped_lut.pkl'

def load_lut(filename=DEFAULT_LUT):
    with open(filename, 'rb') as f:
        return pickle.load(f)
