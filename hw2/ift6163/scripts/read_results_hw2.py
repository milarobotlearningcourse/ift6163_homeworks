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
    eval_avg_return,  eval_std_return = [], []
    train_avg_return, train_std_return = [], []
    # print([e for e in tf.train.summary_iterator(file)])
    for e in tf.train.summary_iterator(file):
        for v in e.summary.value:
            if v.tag == 'Train_EnvstepsSoFar':
                steps.append(v.simple_value)
            elif v.tag == 'Eval_AverageReturn':
                eval_avg_return.append(v.simple_value)
            elif v.tag == 'Eval_StdReturn':
                eval_std_return.append(v.simple_value)
            elif v.tag == 'Train_AverageReturn':
                train_avg_return.append(v.simple_value)
            elif v.tag == 'Train_StdReturn':
                train_std_return.append(v.simple_value)
    return np.array(steps), np.array(eval_avg_return), np.array(eval_std_return), np.array(train_avg_return), np.array(train_std_return)


if __name__ == '__main__':
    logdir = 'data/*/events*'
    eventfiles = glob.glob(logdir)
    print(eventfiles)
    ## Q2
    q2_eventfile = [e for e in eventfiles if "q2" in e][0]
    steps, eval_avg_return, eval_std_return, train_avg_return, train_std_return = get_section_results(q2_eventfile)

    x_pos = np.arange(2)
    CTEs = [train_avg_return[0], eval_avg_return[0]]
    error = [train_std_return[0], eval_std_return[0]]
    materials = ['Train', 'Eval']
    # Build the plot
    fig, ax = plt.subplots()
    ax.bar(x_pos, CTEs, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)
    ax.set_ylabel('Average Reward')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(materials)
    # ax.set_title('Coefficent of Thermal Expansion (CTE) of Three Metals')
    ax.yaxis.grid(True)

    # Save the figure and show
    plt.tight_layout()
    plt.savefig('q2.png')
    plt.show()

    ## Q3
    q3_eventfiles = [e for e in eventfiles if "q3" in e]
    for eventfile in q3_eventfiles:
        steps, eval_avg_return, eval_std_return, train_avg_return, train_std_return = get_section_results(eventfile)
        name = eventfile.split('/')[1]
        env_name = name.split("_")[2]

        plt.plot(steps, eval_avg_return, label="Eval")
        plt.fill_between(steps, eval_avg_return - eval_std_return, eval_avg_return + eval_std_return, alpha=0.2)

        plt.plot(steps, train_avg_return, label="Train")
        plt.fill_between(steps, train_avg_return - train_std_return, train_avg_return + train_std_return, alpha=0.2)

        plt.title(env_name)
        plt.legend(loc="lower right")
        plt.savefig(f'q3_{env_name}.png')
        plt.show()

    ## Q4
    q4_eventfiles = [e for e in eventfiles if "q4" in e]
    ensembles, horizons, numseqs = {}, {}, {}
    for eventfile in q4_eventfiles:
        name = eventfile.split('/')[1]

        steps, returns, stds, _, _ = get_section_results(eventfile)

        if "ensemble" in name:
            ensembles[int(name[23])] = {
                "steps": steps, "returns": returns, "stds": stds
            }

        if "horizon" in name:
            horizons[int(name.split("_")[3][7:])] = {
                "steps": steps, "returns": returns, "stds": stds
            }

        if "numseq" in name:
            numseqs[int(name.split("_")[3][6:])] = {
                "steps": steps, "returns": returns, "stds": stds
            }

    def pp(d, name):
        for k, v in d.items():
            x = v["steps"]
            y = v["returns"]
            s = v["stds"]
            plt.plot(x, y, label=k)
            plt.fill_between(x, y - s, y + s, alpha=0.2)
        plt.legend(loc="lower right")
        plt.savefig(name)
        plt.show()


    pp(ensembles, "q4_ensembles.png")
    pp(horizons, "q4_horizons.png")
    pp(numseqs, "q4_numseqs.png")

    ## Q5s
    q5_eventfiles = [e for e in eventfiles if "q5" in e]
    for eventfile in q5_eventfiles:
        name = eventfile.split('/')[1]
        title = name.split('_')[3]
        if title == "cem":
            title += name.split('_')[4]

        steps, returns, stds, _, _ = get_section_results(eventfile)
        plt.plot(steps, returns, label=title)

    plt.legend(loc="lower right")
    plt.savefig(f'q5.png')
    plt.show()