import numpy as np
from torch import tensor
import pickle
import sys

# adding Folder_2 to the system path
sys.path.insert(0, "..")

from ref.memristor import G_STEPS, V_STEPS

SPICE_WINDOW = 4
SUFFIXES = {
    "p": 1e-12,
    "n": 1e-9,
    "u": 1e-6,
    "m": 1e-3,
}


def get_file_path(r_step):
    return f"../lut/res_r_{r_step}.cir.lis"


def process_line(line):
    val = line.strip().split(" ")[-1]
    last_char = val[-1]

    if last_char in SUFFIXES.keys():
        return float(val[:-1]) * SUFFIXES[last_char]
    else:
        return float(val)


def process_file(input_file):
    with open(input_file, "r") as file:
        lines = file.readlines()

    index = None
    result = []

    for line in lines:
        if index is None:
            if line.strip() == "x":
                index = -1
        else:
            if line.strip() == "y":
                break
            elif index == SPICE_WINDOW:
                result.append(process_line(line))
                index = 0
            else:
                index += 1

    return result


def build_lut():
    r_steps = range(G_STEPS)
    lut = np.zeros((G_STEPS, V_STEPS))

    for r_step in r_steps:
        file_path = get_file_path(r_step)
        data = process_file(file_path)
        lut[r_step] = data

    lut = (lut - np.min(lut)) / (np.max(lut) - np.min(lut))
    return lut


def map_lut(spice_lut):
    # # todo: implement this
    c = V_STEPS
    r = G_STEPS
    # initialize lut
    lut = np.zeros((r, c))

    for c_i in range(c):
        for r_i in range(r):
            negative = False
            # get c_step and r_step from lut
            c_step, r_step = get_step_index_lut(c_i, r_i)

            # check if c_step and r_step multiples in negetive or not
            if c_step < 0 and r_step > 0:
                negative = True
            elif c_step > 0 and r_step < 0:
                negative = True

            c_step = abs(c_step)
            r_step = abs(r_step)

            # get c_index and r_index from spice lut
            c_index, r_index = get_index_spice_lut(c_step, r_step)

            if negative:
                lut[r_i][c_i] = -spice_lut[r_index][c_index]
            else:
                lut[r_i][c_i] = spice_lut[r_index][c_index]

    return lut


def get_index_spice_lut(c_step, r_step):
    c_index = get_index_from_value(c_step, V_STEPS)
    r_index = 256 - get_index_from_value(r_step, G_STEPS) - 1
    return c_index, r_index


def get_step_index_lut(c_index, r_index):
    c_step = get_value_from_range(c_index, V_STEPS)
    r_step = get_value_from_range(r_index, G_STEPS)
    return c_step, r_step


def get_index_from_value(value, total_numbers):
    # Define the range
    start_range = 0
    end_range = 1

    # Calculate the step size (difference between consecutive numbers)
    step_size = (end_range - start_range) / (total_numbers - 1)

    # Calculate the index corresponding to the given value
    index = (value - start_range) / step_size

    # Return the integer part of the index (to handle floating-point rounding)
    return int(round(index))


def get_value_from_range(i, total_numbers):
    # Define the range
    start_range = -1
    end_range = 1

    # Calculate the difference between consecutive numbers
    step_size = (end_range - start_range) / (total_numbers - 1)

    # Calculate the i-th number
    ith_number = start_range + (i) * step_size

    return ith_number


def get_mapped_lut():
    return tensor(map_lut(build_lut()))


if __name__ == "__main__":
    lut = build_lut()
    mlut = map_lut(lut)
    print(mlut)
    with open("mapped_lut.pkl", "wb") as f:
        pickle.dump(mlut, f)
    # with open("mapped_lut.csv", "w", newline="") as csvfile:
    #     writer = csv.writer(csvfile)
    #     for row in mlut:
    #         writer.writerow(row)
