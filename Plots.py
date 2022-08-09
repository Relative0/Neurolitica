import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

sns.set()
from Methods_NN import SHAP_Averaging, Compute_Individual_Predictions

def Choose_Plots(Data):
    # List = ["Summary_Line", "Scatter_Reg", "distplot", "violin", "swarmplot", "strip_point", "lm", "lm_reg",
    # "pairgrid_scatter_kde", "facetgrid", "joint", "kde", "pair"]
    # List = ["Summary_Line"]
    List = [ "pair"]

    Bin_Data = Data["BinData"]
    SHAP_Data = Data["SHAPData"]


    for Choice in List:
        if (Choice == "Summary_Line"):
            scatter_line(Data)

        elif (Choice == "Scatter_Reg"):
            scatter_reg(Data)

        elif (Choice == "Example_Plot"):
            Example_Plot_1(Data)

        elif (Choice == "distplot"):
            distplot(Data)

        elif (Choice == "violin"):
            violin(Data)

        elif (Choice == "swarmplot"):
            swarmplot(Data)

        elif (Choice == "strip_point"):
            strip_point(Data)

        elif (Choice == "lm"):
            lm(Data)

        elif (Choice == "lm_reg"):
            lm_reg(Data)

        elif (Choice == "pairgrid_scatter_kde"):
            pairgrid_scatter_kde(Data)

        elif (Choice == "facetgrid"):
            facetgrid(Data)

        elif (Choice == "joint"):
            joint(Data)

        elif (Choice == "kde"):
            kde(Data)

        elif (Choice == "pair"):
            pair(Data)

        else:
            print("There is an issue in Plotting via Plots.py")

# Here we get multiple data metrics for the set - the SHAP values and individual question prediction metrics.
def Get_Data(Bins, df, question_names):
    X = df.iloc[:, 0:-1]
    Y = df.iloc[:, -1:]

    X_arr = X.values
    Y_arr = Y.values

    Results_DF = pd.DataFrame()

    Metrics_df = pd.DataFrame(columns=X.columns)
    # AveSHAP_df = pd.DataFrame()
    Question_Predictions_Ave, Question_Predictions_Ind = Compute_Individual_Predictions(df, Bins, 10)
    Bincount = 0

    for BinsizeName, TheBinsizeList in Bins:
        NumberofBins = len(TheBinsizeList) + 1
        # Need to filter this by bins now that I have multiple bin data.
        Question_Predictions_Ave_Bin = Question_Predictions_Ave[Question_Predictions_Ave['Binsize'] == NumberofBins]
        Metrics = ["Accuracy", "Precision", "Recall", "F1"]
        for l in Metrics:
            Metric_value = Question_Predictions_Ave_Bin.explode(l).reset_index()
            # Metrics_df = Metrics_df.append(Metric_value[l], columns=list(X.columns))
            # Metrics_df.loc[-1] = Metric_value[l]
            # df2 = df.append(pd.DataFrame([new_row], index=['7'], columns=df.columns))
            # Metrics_df.loc[len(Metrics_df.index)] =  Metric_value[l].tolist()

            Metrics_df.loc[l] = np.array(Metric_value[l].tolist(),dtype=np.float32)
            # Results_DF.append(Metrics_df)

            # Metrics_df.append(Metric_value[l])


        SHAPConfig_Avg, AllConfig_Avg = SHAP_Averaging(X_arr, Y_arr, Bins[Bincount], question_names)
        # Metrics_df.loc["AveSHAP"] = AveSHAP.tolist()

        Combined_df = pd.concat([Metrics_df, AllConfig_Avg, SHAPConfig_Avg], ignore_index=False, sort=True)

        Combined_df_T = Combined_df.transpose()

        # Create a list of ordered question numbers.
        QuestionIterator = list(range(1, len(X.columns)+1))
        Combined_df_T.insert(0, "Question", QuestionIterator, True)

        # Create and give values for the "Bin".
        Combined_df_T.insert(0, "Bin", NumberofBins, True)

        Results_DF = pd.concat([Results_DF, Combined_df_T], ignore_index=False)

    cols = ['Bin']
    Results_DF[cols] = Results_DF[cols].astype('category')
    Bincount = Bincount + 1

        # Results_DF['Bin']= Results_DF['Bin'].apply(str)
        # Results_DF['Question'] = Results_DF['Question'].apply(str)

    return Results_DF

# Ultimately I'll need to pass in a list of vals i.e. Accuracies, F1 etc. and the "Indexes" will be the questions.
def BinData_RandomVals(N):
    Accuracy_Indexes = list(range(1, 5))
    df = pd.DataFrame({'Bin': np.random.randint(1, 10, N)})
    AccuraciesList = []
    # Creates a random list of per-item accuracies.
    for item in range(N):
        AccuraciesList.append([item + 1, np.random.random_sample(size=len(Accuracy_Indexes)).tolist()])

    df = pd.DataFrame.from_records(AccuraciesList, columns=['Bin', 'Accuracies'])

    df_long = df.explode('Accuracies', ignore_index=True)
    df_long['Accuracy_Indexes'] = np.tile(Accuracy_Indexes, len(df))
    df_long['Accuracies'] = df_long['Accuracies'].astype(float)

    # df_long = ScoringMetricsConcat_DF_Subset.explode(l).reset_index()
    # df_long.drop('index', axis=1, inplace=True)
    # df_long['Questions'] = np.tile(QuestionIterator, len(ScoringMetricsConcat_DF_Subset))
    # df_long[l] = df_long[l].astype(float)

    return df_long

def Example_Plot_1(df):
    df_long = df["Bin"]

    # g = sns.catplot(x='Accuracy_Indexes', y='Accuracies', hue="Bin", data=df_long, height=5, aspect=.8)
    # plt.show()
    # plt.figure(figsize=(15, 10))
    ax = sns.scatterplot(x='Accuracy_Indexes', y='Accuracies', hue="Bin",
                         legend='full',
                         data=df_long,
                         palette=sns.color_palette("Set1", n_colors=len(df_long.Bin.unique())))
    Accuracy_Indexes_per_Accuracy = df_long.groupby('Accuracy_Indexes')['Accuracies'].max()
    sns.lineplot(data=Accuracy_Indexes_per_Accuracy,ax=ax.axes,color='black')
    # max_transistors_per_year = df.groupby('Accuracy_Indexes')['Accuracies'].max()
    plt.tight_layout()
    plt.show()

def scatter_line(df):
    df_long = df["Bin"]

    # g = sns.catplot(x='Accuracy_Indexes', y='Accuracies', hue="Bin", data=df_long, height=5, aspect=.8)
    # plt.show()
    # plt.figure(figsize=(15, 10))
    ax = sns.scatterplot(x='Accuracy_Indexes', y='Accuracies', hue="Bin",
                         legend='full',
                         data=df_long,
                         palette=sns.color_palette("Set1", n_colors=len(df_long.Bin.unique())))
    Accuracy_Indexes_per_Accuracy = df_long.groupby('Accuracy_Indexes')['Accuracies'].max()
    sns.lineplot(data=Accuracy_Indexes_per_Accuracy, ax=ax.axes, color='black')
    # max_transistors_per_year = df.groupby('Accuracy_Indexes')['Accuracies'].max()
    plt.tight_layout()
    plt.show()

def scatter_reg(df):
    df_long = df["Bin"]

    # g = sns.catplot(x='Accuracy_Indexes', y='Accuracies', hue="Bin", data=df_long, height=5, aspect=.8)
    # plt.show()
    # plt.figure(figsize=(15, 10))
    ax = sns.scatterplot(x='Accuracy_Indexes', y='Accuracies', hue="Bin",
                         legend='full',
                         data=df_long,
                         palette=sns.color_palette("Set1", n_colors=len(df_long.Bin.unique())))
    ax = sns.regplot(x='Accuracy_Indexes', y='Accuracies', data=df_long,scatter=False, ax=ax.axes,order=3)
    # ax.set_xlim(2006, 2021)
    # ax.set_ylim(0, 70)
    plt.show()

def distplot(df):
    df_long = df["Bin"]

    # sns.distplot(tips_df["total_bill"], bins=9, label="total_bil")
    # sns.distplot(tips_df["tip"], bins=9, label="tip")
    # sns.distplot(tips_df["size"], bins=9, label="size")
    sns.set(style="white", palette="muted", color_codes=True)
    # sns.displot(data=df_long, y="Accuracies", kde=True)
    # sns.displot(data=df_long["Bin"], y=df_long["Accuracies"], kde=True)
    sns.displot(df_long["Bin"], y=df_long["Accuracies"], kde=True)
    plt.show()

def violin(df):
    df_long = BinData_RandomVals(2)

    # g = sns.catplot(x='Accuracy_Indexes', y='Accuracies', hue="Bin", data=df_long, height=5, aspect=.8)
    # plt.show()
    # plt.figure(figsize=(15, 10))
    # ax = sns.scatterplot(x='Accuracy_Indexes', y='Accuracies', hue="Bin",

    plt.figure(figsize=(15, 10))
    sns.set()
    _, ax = plt.subplots(figsize=(10, 7))
    sns.violinplot(x='Accuracy_Indexes', y='Accuracies', hue="Bin",
                   data=df_long,
                   split=True,
                   bw=.5,
                   cut=0.3,
                   linewidth=1,
                   palette=sns.color_palette(['green', 'red']))
    # ax.set(ylim=(0, 700))
    plt.show()

def swarmplot(df):
    df_long = df["Bin"]

    _, ax = plt.subplots(figsize=(21, 9))
    sns.swarmplot(x='Accuracy_Indexes', y='Accuracies', hue="Bin", data=df_long, size=4)
    plt.show()

def strip_point(df):
    df_long = df["Bin"]
    _, ax = plt.subplots(figsize=(10, 7))
    sns.despine(bottom=True, left=True)

    sns.stripplot(x='Accuracy_Indexes', y='Accuracies', hue="Bin",
                  data=df_long, dodge=.5, alpha=.55, zorder=1)

    sns.pointplot(x='Accuracy_Indexes', y='Accuracies', hue="Bin",
                  data=df_long, dodge=.5, join=False, palette="dark",
                  markers="d", scale=.75, ci=None)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title="Bin",
              handletextpad=0, columnspacing=1,
              loc="best", ncol=2, frameon=True)
    plt.show()

def lm(df):
    df_long = df["Bin"]
    sns.lmplot(x='Accuracy_Indexes', y='Accuracies', data=df_long, hue='Bin', col='Bin', col_wrap=3)
    plt.tight_layout()
    plt.show()

def lm_reg(df):
    df_long = df["Bin"]
    ax = sns.lmplot(x='Accuracy_Indexes', y='Accuracies', data=df_long, hue='Bin', fit_reg=False)
    # ax.axes[0, 0].set_xlim((2006, 2021))
    sns.regplot(x='Accuracy_Indexes', y='Accuracies', data=df_long, scatter=False, ax=ax.axes[0, 0], order=3)
    plt.show()

def pairgrid_scatter_kde(df):
    df_long = df["Bin"]
    ax = sns.PairGrid(df_long[['Accuracies', 'Bin']], diag_sharey=False)
    ax.map_upper(sns.scatterplot)
    ax.map_lower(sns.kdeplot, colors="C0")
    ax.map_diag(sns.kdeplot, lw=1, shade=True)
    plt.show()

def facetgrid(df):
    df_long = df["Bin"]
    ax = sns.FacetGrid(df_long, col="Accuracies", row="Bin", margin_titles=True)
    ax.map(plt.hist, "Bin")
    plt.show()

def joint(df):
    df_long = df["Bin"]
    BinData_RandomVals(7)
    sns.jointplot(x='Accuracies', y='Bin', data=df_long,
                  kind="reg", truncate=False, marginal_kws={'bins': 14},
                  color="m", height=7)
    plt.show()

def kde(df):
    df_long = df["Bin"]
    from matplotlib.ticker import MaxNLocator
    plt.figure(figsize=(6, 6))
    ax = sns.kdeplot(x='Accuracies', y='Bin', data=df_long, shade=True)
    ax.axes.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.show()

def pair(Data):
    SHAP_Data = Data["SHAPData"]
    SHAP_Data.reset_index(drop=True,inplace=True)
    Bin_Data  = Data["BinData"]
    # df_long = df["Bin"]

    N = 6
    SHAP_Data_Cols = SHAP_Data.iloc[:, :N]

    # To get enough results I might have to take all metric predictions for each question prediction e.g. 10 accuracies
    # for 10 runs. Or perhaps more than a single bin - but two or 3 bins and make the diagonal be histograms.
    # SHAP_Data_wout_QuestionCol = SHAP_Data_Cols.drop('Bin', axis=1)
    sns.pairplot(SHAP_Data_Cols, hue='Bin',  height=1.5)

    # penguins = sns.load_dataset("penguins")
    # sns.pairplot(penguins)


    #
    # df = sns.load_dataset('tips')
    #
    # # pairplot with hue sex
    # sns.pairplot(df, hue='size')
    plt.show()

    # # df_long = BinData_RandomVals(2)
    # df_long = Bin_Data["Bin"]
    # print(df_long)
    # sns.pairplot(Bin_Data, hue='Bin')
    plt.show()