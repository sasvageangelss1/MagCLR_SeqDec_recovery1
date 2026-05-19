import sys, os
sys.path.insert(0, "C:/Users/hp/Desktop/zz/proj/MagCLR_SeqDec_recovery1")
os.chdir("C:/Users/hp/Desktop/zz/proj/MagCLR_SeqDec_recovery1")

try:
    from scripts.plot_seqdec_trajectories import load_trajectories, plot_global_comparison
    td = load_trajectories()
    print("Data OK, plotting global comparison")
    plot_global_comparison(td)
    print("Done")
except Exception as e:
    import traceback
    traceback.print_exc()
