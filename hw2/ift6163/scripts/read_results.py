import glob
# import tensorflow as tf
import numpy as np
import tensorflow.compat.v1 as tf
import seaborn as sns
import matplotlib.pyplot as plt

def get_section_results(file):
    """
        requires tensorflow==1.12.0
    """
    steps = []
    avg_return = []
    std_return = []
    # print([e for e in tf.train.summary_iterator(file)])
    for e in tf.train.summary_iterator(file):
        # print([v for v in e.summary.value])
        for v in e.summary.value:
            if v.tag == 'Train_EnvstepsSoFar':
                steps.append(v.simple_value)
            elif v.tag == 'Eval_AverageReturn':
                avg_return.append(v.simple_value)
            elif v.tag == 'Eval_StdReturn':
                std_return.append(v.simple_value)
    return steps, avg_return, std_return


if __name__ == '__main__':
    # logdir = 'data/q1_lb_rtg_na_CartPole-v0_13-09-2020_23-32-10/events*'
    # base = "/home/jg/ift6163_homeworks/"
    # exp = [
    #     "hw2/outputs/2022-02-16/04-29-57/data/hw2_q4_reacher_horizon5_reacher-ift6163-v0_16-02-2022_04-29-57/events.out.tfevents.1645003797.ng30604.narval.calcul.quebec"
    # ]
    # logdir = '/home/jg/ift6163_homeworks/hw2/outputs/2022-02-15/00-29-32/data/hw2_q2_obstacles_singleiteration_obstacles-ift6163-v0_15-02-2022_00-29-32/events*'
    logdir = "/home/jg/ift6163_homeworks/hw2/outputs_final/2022-02-16/*/data/*/events*" # q4
    eventfiles = glob.glob(logdir)
    results = {}
    ensembles = {}
    horizons = {}
    numseqs = {}
    for eventfile in eventfiles:
        name = eventfile.split('/')[9]

        steps, returns, stds = get_section_results(eventfile)
        # if "horizon" in name:
        print(f"\n{name}")
        print(steps)
        print(returns)
        print(stds)
        results[eventfile] = {
            "steps":steps, "returns":returns, "stds":stds
        }
        if "ensemble" in name:
            ensembles[int(name[23])] = {
            "steps":steps, "returns":returns, "stds":stds
        }

        if "horizon" in name:
            horizons[int(name.split("_")[3][7:])] = {
            "steps":steps, "returns":returns, "stds":stds
        }

        if "numseq" in name:
            numseqs[int(name.split("_")[3][6:])] = {
            "steps":steps, "returns":returns, "stds":stds
        }
        # for i, (step, avg, std) in enumerate(zip(steps, returns, stds)):
        #     print(f"Iteration {i:d} | Train steps: {int(step):d} | Return: {avg:.1f} | Std: {std:.1f}")
        # break


    def pp(d, name):
        for k, v in d.items():
            x = np.array(v["steps"])
            y = np.array(v["returns"])
            s = np.array(v["stds"])
            plt.plot(x, y, label=k)
            plt.fill_between(x, y - s, y + s, alpha=0.2)
        plt.legend(loc="lower right")
        plt.savefig(name)
        plt.show()

    pp(ensembles, "ensembles.jpg")
    pp(horizons, "horizons.jpg")
    pp(numseqs, "numseqs.jpg")