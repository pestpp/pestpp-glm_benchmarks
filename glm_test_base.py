# TODO: test variance and mean of draws, add chenoliver and test approx and full solution
import os
import shutil
import platform
import numpy as np
import pandas as pd
import platform
import matplotlib.pyplot as plt
import pyemu




bin_path = os.path.join("test_bin")
if "linux" in platform.platform().lower():
    bin_path = os.path.join(bin_path,"linux")
elif "darwin" in platform.platform().lower():
    bin_path = os.path.join(bin_path,"mac")
else:
    bin_path = os.path.join(bin_path,"win")

bin_path = os.path.abspath("test_bin")
os.environ["PATH"] += os.pathsep + bin_path


# case of either appveyor, travis or local
if os.path.exists(os.path.join("pestpp","bin")):
    bin_path = os.path.join("..","..","pestpp","bin")
else:
    bin_path = os.path.join("..","..","..","..","pestpp","bin")

  
if "windows" in platform.platform().lower():
    exe_path = os.path.join(bin_path, "win", "pestpp-glm.exe")
elif "darwin" in platform.platform().lower():
    exe_path = os.path.join(bin_path,  "mac", "pestpp-glm")
else:
    exe_path = os.path.join(bin_path, "linux", "pestpp-glm")


diff_tol = 1.0e-6
port = 4016


def tenpar_base_test():
    """tenpar basic test"""
    
    model_d = "glm_10par_xsec"
    test_d = os.path.join(model_d, "master_basic_test")
    template_d = os.path.join(model_d, "template")
    if not os.path.exists(template_d):
        raise Exception("template_d {0} not found".format(template_d))
    if os.path.exists(test_d):
        shutil.rmtree(test_d)
    # shutil.copytree(template_d, test_d)
    pst_name = os.path.join(template_d, "pest.pst")
    pst = pyemu.Pst(pst_name)
    pst.parameter_data.loc[:,"partrans"] = "log"

    pst.pestpp_options = {}
    pst.pestpp_options["ies_num_reals"] = 10
    pst.control_data.noptmax = 3

    # pst.pestpp_options["ies_verbose_level"] = 3
    pst_name = os.path.join(template_d, "pest_basic.pst")
    pst.write(pst_name)
    pyemu.os_utils.start_workers(template_d, exe_path, "pest_basic.pst", num_workers=10,
                               master_dir=test_d, verbose=True, worker_root=model_d,
                               port=port)


if __name__ == "__main__":  
    tenpar_base_test()

    