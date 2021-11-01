"""

Author: diegofg293@gmail.com
Date: 25-10-2021

Description: A Library with functions to perform a churn model

"""

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, plot_roc_curve
from sklearn.model_selection import GridSearchCV, train_test_split

from churn_library_solution.conf import config, constants
from churn_library_solution.conf.config import logger

sns.set()


def import_data(pth):
    """
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    """
    df = pd.read_csv(r"./data/bank_data.csv")
    return df


def perform_eda(df):
    """
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    """

    plt.figure(figsize=(20, 10))
    df["Churn"].hist()
    plt.tight_layout()
    plt.savefig(config.EDA_IMAGES / "churn_distribution.png")
    plt.close()

    plt.figure(figsize=(20, 10))
    df["Customer_Age"].hist()
    plt.tight_layout()
    plt.savefig(config.EDA_IMAGES / "customer_age_distribution.png")
    plt.close()

    plt.figure(figsize=(20, 10))
    df.Marital_Status.value_counts("normalize").plot(kind="bar")
    plt.tight_layout()
    plt.savefig(config.EDA_IMAGES / "marital_status_distribution.png")
    plt.close()

    plt.figure(figsize=(20, 10))
    sns.distplot(df["Total_Trans_Ct"])
    plt.tight_layout()
    plt.savefig(config.EDA_IMAGES / "total_transaction_distribution.png")
    plt.close()

    plt.figure(figsize=(20, 10))
    sns.heatmap(df.corr(), annot=False, cmap="Dark2_r", linewidths=2)
    plt.tight_layout()
    plt.savefig(config.EDA_IMAGES / "heatmap.png", bbox_inches="tight")


def encoder_helper(df, category_lst, response):
    """
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    """
    for col in category_lst:
        new_lst = []
        group_obj = df.groupby(col).mean()[response]

        for val in df[col]:
            new_lst.append(group_obj.loc[val])

        new_col_name = col + "_" + response
        df[new_col_name] = new_lst

    return df


def perform_feature_engineering(df, response):
    """
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    """
    y = df["Churn"]
    X = pd.DataFrame()
    df = encoder_helper(df, constants.CAT_COLUMNS, response)

    X[constants.KEEP_COLS] = df[constants.KEEP_COLS]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test


def classification_report_image(
    y_train, y_test, y_train_preds_lr, y_train_preds_rf, y_test_preds_lr, y_test_preds_rf
):
    """
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    """
    plt.figure()
    plt.rc("figure", figsize=(8, 8))
    plt.text(
        0.01,
        0.05,
        str(classification_report(y_test, y_test_preds_rf)),
        {"fontsize": 10},
        fontproperties="monospace",
    )
    plt.text(
        0.01,
        0.6,
        "Random Forest Test (below) Random Forest Train (above)",
        {"fontsize": 10},
        fontproperties="monospace",
    )
    plt.text(
        0.01,
        0.7,
        str(classification_report(y_train, y_train_preds_rf)),
        {"fontsize": 10},
        fontproperties="monospace",
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(config.RESULTS_IMAGES / "rf_results.png")
    plt.close()

    plt.figure()
    plt.rc("figure", figsize=(8, 8))
    plt.text(
        0.01,
        0.05,
        str(classification_report(y_train, y_train_preds_lr)),
        {"fontsize": 10},
        fontproperties="monospace",
    )
    plt.text(
        0.01,
        0.6,
        "Logistic Regression Test (below) Logistic Regression Train (above)",
        {"fontsize": 10},
        fontproperties="monospace",
    )
    plt.text(
        0.01,
        0.7,
        str(classification_report(y_test, y_test_preds_lr)),
        {"fontsize": 10},
        fontproperties="monospace",
    )  # approach improved by OP -> monospace!
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(config.RESULTS_IMAGES / "logistic_results.png")
    plt.close()


def feature_importance_plot(model, X_data, output_pth):
    """
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    """
    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel("Importance")

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)
    plt.tight_layout()
    plt.savefig(output_pth)
    plt.close()


def train_models(X_train, X_test, y_train, y_test):
    """
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    """
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression()
    param_grid = {
        "n_estimators": [200, 500],
        "max_features": ["auto", "sqrt"],
        "max_depth": [4, 5, 100],
        "criterion": ["gini", "entropy"],
    }
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    lrc.fit(X_train, y_train)
    # predict on train data
    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)
    # predict on test data
    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    # scores
    logger.info("random forest results")
    logger.info("test results")
    logger.info(classification_report(y_test, y_test_preds_rf))
    logger.info("train results")
    logger.info(classification_report(y_train, y_train_preds_rf))
    logger.info("logistic regression results")
    logger.info("test results")
    logger.info(classification_report(y_test, y_test_preds_lr))
    logger.info("train results")
    logger.info(classification_report(y_train, y_train_preds_lr))

    # plots
    # roc auc
    lrc_plot = plot_roc_curve(lrc, X_test, y_test)
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    # rfc_disp
    plot_roc_curve(cv_rfc.best_estimator_, X_test, y_test, ax=ax, alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)
    config.RESULTS_IMAGES
    plt.savefig(config.RESULTS_IMAGES / "roc_curve_result.png")
    plt.close()

    # save best model
    joblib.dump(cv_rfc.best_estimator_, config.MODEL_DIR / "rfc_model.pkl")
    joblib.dump(lrc, config.MODEL_DIR / "logistic_model.pkl")

    # save model results
    classification_report_image(
        y_train, y_test, y_train_preds_lr, y_train_preds_rf, y_test_preds_lr, y_test_preds_rf
    )

    # store feature importances plot
    feature_importance_plot(
        cv_rfc.best_estimator_, X_train, config.RESULTS_IMAGES / "feature_importances.png"
    )
