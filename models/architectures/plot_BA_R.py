import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np
import scipy


def regression_plot(sbpTrues, sbpPreds, dbpTrues, dbpPreds, mapTrues, mapPreds):
    # plt.figure(figsize=(18, 6), dpi=120)
    plt.figure(figsize=(15, 8), dpi=120)

    plt.subplot(1, 3, 1)
    sns.regplot(x=sbpTrues, y=sbpPreds, scatter_kws={'alpha': 0.8, 's': 3}, line_kws={'color': 'r', 'lw': 3})
    plt.xlabel('Target Value (mmHg)', fontsize=14)
    plt.ylabel('Estimated Value (mmHg)', fontsize=14)
    plt.title('SBP', fontsize=18)
    # plt.show()

    # plt.figure(figsize=(8, 5), dpi=120)
    plt.subplot(1, 3, 2)
    sns.regplot(x=dbpTrues, y=dbpPreds, scatter_kws={'alpha': 0.8, 's': 3}, line_kws={'color': 'r', 'lw': 3})
    plt.xlabel('Target Value (mmHg)', fontsize=14)
    plt.ylabel('Estimated Value (mmHg)', fontsize=14)
    plt.title('DBP', fontsize=18)
    # plt.show()

    # plt.figure(figsize=(8, 5), dpi=120)
    plt.subplot(1, 3, 3)
    # sns.regplot(x=mapTrues, y=mapPreds, scatter_kws={'alpha': 0.5, 's': 3}, line_kws={'color': '#e0b0b4'})
    sns.regplot(x=mapTrues, y=mapPreds, scatter_kws={'alpha': 0.8, 's': 3}, line_kws={'color': 'r', 'lw': 3})
    plt.xlabel('Target Value (mmHg)', fontsize=14)
    plt.ylabel('Estimated Value (mmHg)', fontsize=14)
    plt.title('MAP', fontsize=18)

    plt.suptitle('Regression Plot', fontsize=18)
    # plt.figure(figsize=(8, 5), dpi=120)

    plt.show()

    '''
        Printing statistical analysis values like r and p value
    '''
    print('DBP')
    print(scipy.stats.linregress(dbpTrues, dbpPreds))
    print('MAP')
    print(scipy.stats.linregress(mapTrues, mapPreds))
    print('SBP')
    print(scipy.stats.linregress(sbpTrues, sbpPreds))



def bland_altman_plot(SBP_predict, SBP_actual, DBP_predict, DBP_actual, MAP_predict, MAP_actual):
    def bland_altman(data1, data2, x_axis=True):
        """
        Computes mean +- 1.96 sd

        Arguments:
            data1 {array} -- series 1
            data2 {array} -- series 2
        """

        data1 = np.asarray(data1)
        data2 = np.asarray(data2)
        mean = np.mean([data1, data2], axis=0)
        diff = data1 - data2  # Difference between data1 and data2
        md = np.mean(diff)  # Mean of the difference
        sd = np.std(diff, axis=0)  # Standard deviation of the difference

        plt.scatter(mean, diff, alpha=0.6, s=11)
        plt.axhline(md, color='black', linestyle='--', alpha=1)
        plt.axhline(md + 1.96 * sd, color='black', linestyle='--', alpha=1)
        plt.axhline(md - 1.96 * sd, color='black', linestyle='--', alpha=1)
        plt.text(x=np.min(mean), y=md + 3 * sd, s='{}'.format(round(md + 1.96 * sd, 2)), fontsize=10)
        plt.text(x=np.min(mean), y=md - 3 * sd, s='{}'.format(round(md - 1.96 * sd, 2)), fontsize=10)

        max = int(3 * (md + 1.96 * sd))
        min = int(3 * (md - 1.96 * sd))
        # plt.ylim(ymin=-30, ymax=30)
        # plt.ylim(ymin=-55, ymax=55)
        plt.ylim(ymin=-85, ymax=85)
        if x_axis:
            plt.xlabel('Avg. of Target and Estimated Value (mmHg)', fontsize=14)
        plt.ylabel('Error in Prediction (mmHg)', fontsize=14)
        print(md + 1.96 * sd, md - 1.96 * sd)
        ind = np.where(np.logical_and(md - 1.96 * sd < diff, diff < md + 1.96 * sd))
        print('Per point in LOAs: {}%'.format(round(len(ind[0]) / len(diff) * 100, 2)))

    plt.subplot(3, 1, 1)
    bland_altman(SBP_predict, SBP_actual, False)
    plt.title('SBP', fontsize=15)
    # plt.suptitle('Bland-Altman Plot', fontsize=18)
    # plt.gcf().set_size_inches(8, 5)
    # plt.show()

    plt.subplot(3, 1, 2)
    bland_altman(DBP_predict, DBP_actual, False)
    plt.title('DBP', fontsize=15)
    # plt.suptitle('Bland-Altman Plot', fontsize=18)
    # plt.gcf().set_size_inches(8, 5)

    # plt.show()

    plt.subplot(3, 1, 3)
    bland_altman(MAP_predict, MAP_actual)
    plt.title('MAP', fontsize=15)

    plt.suptitle('Bland-Altman Plot', fontsize=18)
    plt.gcf().set_size_inches(15, 8)
    plt.show()
    plt.close()