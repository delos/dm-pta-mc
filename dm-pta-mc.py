"""
    Main program
"""

import numpy as np
from mpi4py import MPI
import sys
from time import time

from src.os_util import my_mkdir
from src.input_parser import get_input_variables, get_c_list, read_halos
import src.parallel_util as put
from src.snr import opt_pulsar_snr
import src.signals as signals
import src.snr as snr
import src.generate_sim_quants as gsq
import src.constants as const
import src.profile as profile
import src.phase_shift_integrals as integrals

#####

# initializing MPI
comm = MPI.COMM_WORLD

# get the total number of processors
n_proc = comm.Get_size()

# get the id of processor
proc_id = comm.Get_rank()

# ID of main processor
root_process = 0

# get start time
time_start = time()

if proc_id == root_process:

    print("--- DM - PTA - MC ---")
    print()
    print("    v1.0")
    print()
    print("    Running on " + str(n_proc) + " processors")
    print()
    print("---")
    print()
    print("Reading input file...")
    print()
    sys.stdout.flush()

in_filename = sys.argv[-1]
in_dict = get_input_variables(in_filename)

if proc_id == root_process:

    print("Done reading input file!")
    print()
    print("    Input variables:")

    for key in in_dict:

        print("        " + key + " : " + str(in_dict[key]))

    print()
    print("---")
    print()
    sys.stdout.flush()

# tell ehich processors what to compute for
job_list = None
job_list_recv = None

if proc_id == root_process:

    # number of jobs to do
    num_jobs = in_dict["NUM_UNIVERSE"]
    total_job_list = []

    for ui in range(num_jobs):
        total_job_list.append([ui])

    job_list = put.generate_job_list(n_proc, np.array(total_job_list))

job_list_recv = comm.scatter(job_list, root=root_process)

if proc_id == root_process:

    print("    Determining simulation variables...")
    print()
    sys.stdout.flush()

dt = const.week_to_s * in_dict["DT_WEEK"]
obs_T = const.yr_to_s * in_dict["T_YR"]

# time limit for run
if in_dict["TIMELIMIT_HOUR"] > 0:
  time_end = time_start + in_dict["TIMELIMIT_HOUR"]*3600.
else:
  time_end = np.inf

# random seed
try:
    # set seed deterministically
    seed = int(in_dict["SEED"])
except:
    # set seed randomly
    # probably not needed, but here just in case
    # different processes start in the same state
    seed = np.frombuffer(np.random.bytes(4), dtype='>u4')[0]
np.random.seed(seed + proc_id)

# number of time points
Nt = int(obs_T / dt)
t_grid = np.linspace(0, dt * Nt, num=Nt, endpoint=False)
t_grid_yr = t_grid / const.yr_to_s

v_bar = const.km_s_to_kpc_yr * in_dict["V_BAR_KM_PER_SEC"]
v_0 = const.km_s_to_kpc_yr * in_dict["V_0_KM_PER_SEC"]
v_E = const.km_s_to_kpc_yr * in_dict["V_E_KM_PER_SEC"]
v_Esc = const.km_s_to_kpc_yr * in_dict["V_ESC_KM_PER_SEC"]
max_R = in_dict["R_FACTOR"] * v_bar * t_grid_yr[-1]

if proc_id == root_process:
    verbose = True
else:
    verbose = False

[num_objects, max_R, log10_M_min] = gsq.set_num_objects(
    max_R,
    log10_f=in_dict["LOG10_F"],
    log10_M=in_dict["LOG10_M"],
    use_HMF=in_dict["USE_HMF"],
    HMF_path=in_dict["HMF_PATH"],
    log10_M_min=in_dict["LOG10_M_MIN"],
    min_num_object=in_dict["MIN_NUM_OBJECT"],
    number_density=in_dict["N_MSUN_PER_KPC3"],
    verbose=verbose,
)

# 
if in_dict["USE_EXACT_DPHI"]:
    if len(in_dict["FORM_PATH"]) > 0:
        form_x, form_F = profile.form_table(in_dict["FORM_PATH"])
        form_power = in_dict["MASS_POWER"]
    else:
        form_x, form_F = profile.form_table()
        form_power = 2.
    interp_table = integrals.dphi_interpolation_table(form_x, form_F, form_power)
else:
    interp_table = None
        

# custom density profiles
if in_dict["USE_FORMTAB"]:
    form_fun = profile.prepare_form(in_dict["FORM_PATH"])

# list of halos
if in_dict["USE_HALOLIST"]:
    rhos_full, rs_full, v_full = read_halos(in_dict["HALO_PATH"])
    
    if v_full is not None:
        v_full *= const.km_s_to_kpc_yr
    
    if not in_dict["USE_FORMTAB"]:
        raise ValueError('USE_HALOLIST currently requires custom density profile ("USE_FORMTAB")')

# generate positions of pulsars (same across all universes)
dhat_list = gsq.gen_dhats(in_dict["NUM_PULSAR"])

if proc_id == root_process:

    print("    Random seed                            = " + str(seed))
    print("    Number of time points                  = " + str(Nt))
    print("    Number of subhalos per pulsar/earth    = " + str(num_objects))
    print("    Radius of simulation sphere            = " + str(max_R) + " kpc")
    print()

    if in_dict["USE_HMF"]:

        print("    Halo mass function M_min = " + str(10 ** log10_M_min) + " M_sol")
        print()

    print("---")
    print()
    sys.stdout.flush()

snr_list = []

for job in range(len(job_list_recv)):

    if job_list_recv[job, 0] != -1:

        try:

            uni_id = job_list_recv[job, 0]

            snr_list_uni = []

            if in_dict["CALC_TYPE"] == "pulsar":

                if proc_id == root_process and job == 0:

                    print("Starting PULSAR term calculation...")
                    print()
                    print("    Generating signals and computing optimal pulsar SNR...")
                    sys.stdout.flush()

                for pul in range(in_dict["NUM_PULSAR"]):
                    
                    if time() > time_end: raise TimeoutError

                    r0_list = gsq.gen_positions(max_R, num_objects)

                    if in_dict["USE_HALOLIST"]:

                        rhos_list, rs_list, v_list = gsq.sample_halos(rhos_full, rs_full, v_full, num_objects)

                        profile_list = {'rs':rs_list, 'rhos':rhos_list}
                        
                        if v_list is None:
                            
                            v_list = gsq.gen_velocities(v_0, v_Esc, v_E, num_objects)

                    else:

                        v_list = gsq.gen_velocities(v_0, v_Esc, v_E, num_objects)

                        mass_list = gsq.gen_masses(
                            num_objects,
                            use_HMF=in_dict["USE_HMF"],
                            log10_M=in_dict["LOG10_M"],
                            HMF_path=in_dict["HMF_PATH"],
                            log10_M_min=log10_M_min,
                        )

                        conc_list = get_c_list(
                            mass_list,
                            in_dict["USE_FORM"],
                            in_dict["USE_CM"],
                            c=in_dict["C"],
                            cM_path=in_dict["CM_PATH"],
                        )

                        profile_list = {'M':mass_list, 'c':conc_list}

                    d_hat = dhat_list[pul]

                    dphi = signals.dphi_dop_chunked(
                        t_grid_yr,
                        profile_list,
                        r0_list,
                        v_list,
                        d_hat,
                        use_form=in_dict["USE_FORM"],
                        use_chunk=in_dict["USE_CHUNK"],
                        chunk_size=in_dict["CHUNK_SIZE"],
                        form_fun=form_fun,
                        interp_table=interp_table,
                        time_end=time_end,
                    )

                    ht = signals.subtract_signal(t_grid, dphi)

                    snr_val = snr.opt_pulsar_snr(
                        ht, in_dict["T_RMS_NS"], in_dict["DT_WEEK"]
                    )

                    snr_list_uni.append([uni_id, pul, snr_val])

                if proc_id == root_process and job == len(job_list_recv) - 1:
                    print("    Done computing SNR!")
                    print()
                    print("Returning data to main processor...")
                    print()
                    sys.stdout.flush()

            if in_dict["CALC_TYPE"] == "earth":

                if proc_id == root_process:

                    print("Starting EARTH term calculation...")
                    print()
                    print("    Generating signals and computing optimal Earth SNR...")
                    sys.stdout.flush()

                r0_list = gsq.gen_positions(max_R, num_objects)

                if in_dict["USE_HALOLIST"]:

                    rhos_list, rs_list, v_list = gsq.sample_halos(rhos_full, rs_full, v_full, num_objects)

                    profile_list = {'rs':rs_list, 'rhos':rhos_list}
                        
                    if v_list is None:
                        
                        v_list = gsq.gen_velocities(v_0, v_Esc, v_E, num_objects)

                else:

                    v_list = gsq.gen_velocities(v_0, v_Esc, v_E, num_objects)

                    mass_list = gsq.gen_masses(
                        num_objects,
                        use_HMF=in_dict["USE_HMF"],
                        log10_M=in_dict["LOG10_M"],
                        HMF_path=in_dict["HMF_PATH"],
                        log10_M_min=log10_M_min,
                    )

                    conc_list = get_c_list(
                        mass_list,
                        in_dict["USE_FORM"],
                        in_dict["USE_CM"],
                        c=in_dict["C"],
                        cM_path=in_dict["CM_PATH"],
                    )

                    profile_list = {'M':mass_list, 'c':conc_list}

                dphi_vec = signals.dphi_dop_chunked_vec(
                    t_grid_yr,
                    profile_list,
                    r0_list,
                    v_list,
                    use_form=in_dict["USE_FORM"],
                    use_chunk=in_dict["USE_CHUNK"],
                    chunk_size=in_dict["CHUNK_SIZE"],
                    form_fun=form_fun,
                    interp_table=interp_table,
                    time_end=time_end,
                )  # (Nt, 3)

                ht_list = np.zeros((in_dict["NUM_PULSAR"], Nt))

                for pul in range(in_dict["NUM_PULSAR"]):
                    
                    if time() > time_end: raise TimeoutError

                    d_hat = dhat_list[pul]

                    dphi = np.einsum("ij,j->i", dphi_vec, d_hat)

                    ht = signals.subtract_signal(t_grid, dphi)
                    ht_list[pul, :] = ht

                snr_val = snr.opt_earth_snr(
                    ht_list, in_dict["T_RMS_NS"], in_dict["DT_WEEK"]
                )

                snr_list_uni.append([uni_id, -1, snr_val])

                if proc_id == root_process and job == len(job_list_recv) - 1:
                    print("    Done computing SNR!")
                    print()
                    print("Returning data to main processor...")
                    print()
                    sys.stdout.flush()

            snr_list.extend(snr_list_uni)

        except TimeoutError:
            print("Process %d exceeded timelimit (t=%.2fh): finishing"%(proc_id,(time()-time_start)/3600.))
            sys.stdout.flush()
            break

# return data back to root
all_snr_list = comm.gather(snr_list, root=root_process)

# write to output file
if proc_id == root_process:

    output_file = in_dict["OUTPUT_DIR"] + "/snr_" + in_dict["CALC_TYPE"] + "_" + in_dict["RUN_DESCRIP"] + ".txt"
    print("Done returning data!")
    print()
    print("Writing data to output file:")
    print(output_file)

    my_mkdir(in_dict["OUTPUT_DIR"])

    file = open(output_file,"w",)
        
    for key in in_dict:

        file.write("# " + key + " : " + str(in_dict[key]) + '\n')

    for i in range(n_proc):
        for j in range(len(all_snr_list[i])):

            # universe_index = universe_index_list[int(all_A_stat_list[i][j][0])]
            snr_final = all_snr_list[i][j][1]

            file.write(
                str(int(all_snr_list[i][j][0]))
                + " , "
                + str(all_snr_list[i][j][1])
                + " , "
                + str(all_snr_list[i][j][2])
            )
            file.write("\n")

    file.close()

    print("Done writing data!")
    print("---")
    print()
