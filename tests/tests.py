import pycoustic as pc

log1 = pc.Log(path="Excel Data.xlsx", manufacturer="B&K")

surv = pc.Survey()
surv.add_log(data=log1, name="Position 1")

surv.resi_summary()