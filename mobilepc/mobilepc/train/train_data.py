from preprocess.preprocess_data import CFixedValues, OutlierFixTransformer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC


class MobilePricePipeline:
    """
    Custom class to handle mobile price data with preprocessing and modeling.

    Parameters:
    -----------
    seed_model : int
        Seed for reproducibility.
    numerical_vars : list
        List of numerical variables in the dataset.
    numerical_vars_cad : list
        List of numerical variables that require customized preprocessing.
    categorical_vars : list
        List of categorical variables in the dataset.
    categorical_vars_oh : list
        List of categorical variables for one-hot encoding.

    Attributes:
    -----------
    SEED_MODEL : int
        Seed for reproducibility.
    NUMERICAL_VARS : list
        List of numerical variables in the dataset.
    NUMERICAL_VARS_CAD : list
        List of numerical variables that require customized preprocessing.
    CATEGORICAL_VARS : list
        List of categorical variables in the dataset.
    CATEGORICAL_VARS_OH : list
        List of categorical variables for one-hot encoding.
    PIPELINE : sklearn.pipeline.Pipeline
        Preprocessing and modeling pipeline.

    Methods:
    --------
    create_pipeline()
        Create the preprocessing and modeling pipeline.

    fit_logistic_regression(X_train, y_train)
        Fit the pipeline to the training data.

    """

    def __init__(self, seed_model, numerical_vars, numerical_vars_cad, categorical_vars_oh):
        """
        Initialize the MobilePriceData instance.

        Parameters:
        -----------
        seed_model : int
            Seed for reproducibility.
        numerical_vars : list
            List of numerical variables in the dataset.
        numerical_vars_cad : list
            List of numerical variables that require customized preprocessing.
        categorical_vars : list
            List of categorical variables in the dataset.
        categorical_vars_oh : list
            List of categorical variables for one-hot encoding.
        """
        self.SEED_MODEL = seed_model
        self.NUMERICAL_VARS = numerical_vars
        self.NUMERICAL_VARS_CAD = numerical_vars_cad
        self.CATEGORICAL_VARS_OH = categorical_vars_oh

    def create_pipeline(self):
        """
        Create the preprocessing and modeling pipeline.
        """
        numtr = Pipeline(steps=[
            ('imput_mode', SimpleImputer(strategy='most_frequent')),
            ('outliers', OutlierFixTransformer()),
            ('scaler', StandardScaler())])

        sp_num = Pipeline(steps=[
            ('imput_mode', SimpleImputer(strategy='most_frequent')),
            ('outliers', CFixedValues()),
            ('scaler', StandardScaler())])

        cattr = Pipeline(steps=[('imput_zero', SimpleImputer(strategy='most_frequent')),
                                ('onehot', OneHotEncoder(dtype='int', handle_unknown='error'))])

        coltr = ColumnTransformer(transformers=[
            ('numtr', numtr, [
             x for x in self.NUMERICAL_VARS if x not in self.NUMERICAL_VARS_CAD]),
            ('sp_num', sp_num, self.NUMERICAL_VARS_CAD),
            ('cat', cattr, self.CATEGORICAL_VARS_OH)],
            remainder='passthrough')

        pipeline = Pipeline(steps=[
            ('coltr', coltr),
            ('SVM', SVC(kernel='linear', gamma=0.0001, degree=4, C=42, random_state=self.SEED_MODEL))])

        return pipeline

    def fit_SVC(self, X_train, y_train):
        """
        Fit the preprocessing and modeling pipeline to the training data.

        Parameters:
        -----------
        X_train : array-like
            Training feature data.
        y_train : array-like
            Training target data.

        Returns:
        -------
        model_pipeline : sklearn.pipeline.Pipeline
            Fitted preprocessing and modeling pipeline.
        """
        model_pipeline = self.create_pipeline()
        model_pipeline.fit(X_train, y_train)
        return model_pipeline
