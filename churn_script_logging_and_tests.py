import os

import churn_library_solution as cls
from churn_library_solution.conf import config, constants
from churn_library_solution.conf.config import logger


def test_import(import_data):
    """
    test data import - this example is completed for you to assist with the other test functions
    """
    try:
        df = import_data("./data/bank_data.csv")
        logger.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logger.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logger.error("Testing import_data: The file doesn't appear to have rows and columns")
        raise err

    return df


def test_eda(perform_eda, df):
    """
    test perform eda function
    """
    df["Churn"] = df["Attrition_Flag"].apply(lambda val: 0 if val == "Existing Customer" else 1)
    perform_eda(df)
    path_eda_resulst = config.EDA_IMAGES

    try:
        eda_files = {
            "total_transaction_distribution.png",
            "customer_age_distribution.png",
            "churn_distribution.png",
            "marital_status_distribution.png",
            "heatmap.png",
        }
        eda_files_in_path = set(os.listdir(path_eda_resulst))
        files_difference = eda_files.difference(eda_files_in_path)

        assert len(eda_files_in_path) > 0
        logger.info(f"EDA was saved in {str(config.EDA_IMAGES)}")
    except AssertionError as error:
        logger.warning("The EDA analysis couldn't be done")
        logger.warning(f"Files could not be found: {files_difference}")
        raise error


def test_encoder_helper(encoder_helper, df):
    """
    test encoder helper
    """
    df = encoder_helper(df, constants.CAT_COLUMNS, "Churn")
    try:
        for col in constants.CAT_COLUMNS:
            assert col in df.columns
        logger.info("Testing encoder_helper: SUCCESS")
    except AssertionError as error:
        logger.error("Testing encoder_helper: The enconder could not compute the variables")
        return error

    return df


def test_perform_feature_engineering(perform_feature_engineering, df):
    """
    test perform_feature_engineering
    """
    X_train, X_test, y_train, y_test = cls.perform_feature_engineering(df, "Churn")
    try:
        assert X_train.shape[0] > 0
        assert X_test.shape[0] > 0
        assert len(y_train) > 0
        assert len(y_test) > 0
        assert len(y_train) == len(X_train)
        assert len(y_test) == len(X_test)
        logger.info("Testing perform_feature_engineering: SUCCESS")
    except AssertionError as error:
        logger.error(
            "Testing perform_feature_engineering: "
            "The four objects that should be returned were not."
        )
        raise error

    return X_train, X_test, y_train, y_test


def test_train_models(train_models, X_train, X_test, y_train, y_test):
    """
    test train_models
    """
    train_models(X_train, X_test, y_train, y_test)

    try:
        RESULT_IMAGES_EXPECTED = {
            "feature_importances.png",
            "logistic_results.png",
            "rf_results.png",
            "roc_curve_result.png",
        }
        results_images = set(os.listdir(config.RESULTS_IMAGES))
        assert RESULT_IMAGES_EXPECTED.issubset(results_images)

    except FileNotFoundError as error:
        logger.error("Testing train_models: One or more images could not be found")
        raise error

    try:
        RESULTS_MODELS_EXPECTED = {"logistic_model.pkl", "rfc_model.pkl"}
        results_models = set(os.listdir(config.MODEL_DIR))
        assert results_models.issubset(RESULTS_MODELS_EXPECTED)

    except FileNotFoundError as error:
        logger.error("Testing train_models: One or more models could not be found")
        raise error


if __name__ == "__main__":
    df = test_import(cls.import_data)
    test_eda(cls.perform_eda, df)
    df = test_encoder_helper(cls.encoder_helper, df)
    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = test_perform_feature_engineering(
        cls.perform_feature_engineering, df
    )
    test_train_models(cls.train_models, X_TRAIN, X_TEST, Y_TRAIN, Y_TEST)
