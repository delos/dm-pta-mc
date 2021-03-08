"""
    Functions which handle parsing the input files
"""

import numpy as np

int_params = ["NUM_UNIVERSE", "NUM_PULSAR", "MIN_NUM_OBJECT", "CHUNK_SIZE"]
float_params = [
        "LOG10_F",
        "LOG10_M",
        "T_YR",
        "DT_WEEK",
        "T_RMS_NS",
        "R_FACTOR",
        "LOG10_M_MIN",
        "C",
        "PERCENT_CLOSEST",
        "V_0_KM_PER_SEC",
        "V_ESC_KM_PER_SEC",
        "V_E_KM_PER_SEC",
        "V_BAR_KM_PER_SEC",
        "N_MSUN_PER_KPC3",
        "MASS_POWER",
        "TIMELIMIT_HOUR",
        ]
bool_params = ["USE_HMF", "USE_FORM", "USE_CM", "USE_CLOSEST", "USE_CHUNK", "USE_HALOLIST", "USE_FORMTAB","USE_EXACT_DPHI"]
str_params = ["HMF_PATH","CM_PATH","OUTPUT_DIR","CALC_TYPE","RUN_DESCRIP","FORM_PATH","HALO_PATH","SEED"]


def get_c_list(mass_list, use_form, use_cM, c=100, cM_path=""):
    """

    Reads the concentration parameters from the input file

    """

    num_object = len(mass_list)

    if use_form == True:

        if use_cM == True:

            # interpolate the cM table to get the concentration parameter for each M
            log10_m_table = np.log10(np.loadtxt(cM_path, delimiter=",")[:, 0])
            log10_c_table = np.log10(np.loadtxt(cM_path, delimiter=",")[:, 1])
            c_list = 10 ** np.interp(np.log10(mass_list), log10_m_table, log10_c_table)

        else:

            # all halos have the same concentration parameter
            c_list = np.ones(num_object) * c

    else:

        c_list = np.ones(num_object)

    return c_list

def read_halos(path):
    """

    Reads a list of halos with columns:
    rho_s (Msun/kpc^3), r_s (kpc), v (km/s)

    """

    data = np.loadtxt(path)
    return data[:,0], data[:,1], data[:,2]

def read_form(path):
    """

    Reads form factor table with columns:
    x/r_s, M*F(x)/(rho_s*r_s^3)
    
    """

    data = np.loadtxt(path)
    return data[:,0], data[:,1]


def get_input_variables(filename):
    """

    Given the input file returns a dict containing all the input variables

    """

    in_file = open(filename)

    params = {}
    for line in in_file:
        line = line.strip()
        key_value = line.split("=")
        if not line.startswith("#"):
            if len(key_value) == 2:
                params[key_value[0].strip()] = key_value[1].strip()

    in_dict = {}

    for key in int_params:
        try:
            in_dict[key] = to_int(params[key])
        except KeyError:
            in_dict[key] = 0

    for key in float_params:
        try:
            in_dict[key] = to_float(params[key])
        except KeyError:
            in_dict[key] = 0.
    
    for key in bool_params:
        try:
            in_dict[key] = to_bool(params[key])
        except KeyError:
            in_dict[key] = False

    for key in str_params:
        try:
            in_dict[key] = params[key]
        except KeyError:
            in_dict[key] = ""

    return in_dict


def to_bool(string):

    # return True if string == 'True', return False if string == 'False'. Raise ValueError otherwise
    if string == "True":
        return True
    elif string == "False":
        return False
    else:
        raise ValueError("Boolean parameters have to be either True or False")


def to_float(string):

    # return None if string = 'None', return float(string) otherwise
    if string == "None":
        return None
    else:
        return float(string)


def to_int(string):

    # return None if string = 'None', return int(string) otherwise
    if string == "None":
        return None
    else:
        return int(string)
