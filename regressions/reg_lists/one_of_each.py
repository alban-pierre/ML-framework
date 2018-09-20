regressions = [

    ("Linear Regression", LinearRegression()), 
    ("Ridge", Ridge(alpha = .5)),
    ("RidgeCV", RidgeCV(alphas=[0.1, 1.0, 10.0])),
    ("Lasso", Lasso(alpha = 0.1)),
    ("LassoCV", LassoCV(alphas=[0.1, 1.0, 10.0])),
    ("LassoLars", LassoLars(alpha = 0.1)),
    ("LassoLarsCV", LassoLarsCV()),
    ("LassoLarsIC", LassoLarsIC()),
    ("ElasticNet", ElasticNet(alpha=1.0, l1_ratio=0.5)),
    ("ElasticNetCV", ElasticNetCV(alphas=[0.1, 1.0, 10.0], l1_ratio=[0.1, 0.5, 0.9])),
    ("Lars", Lars()),
    ("LarsCV", LarsCV()),
    ("Orthogonal Matching Pursuit", OrthogonalMatchingPursuit(n_nonzero_coefs = 1)),
    ("Bayesian Ridge", BayesianRidge()),
    ("ARD Regression", ARDRegression()),
    ("SGD Regression", SGDRegressor()),
    ("Passive Aggressive Regression", PassiveAggressiveRegressor()),
    ("RANSAC Regression", RANSACRegressor()),
    ("Theil Sen Regression", TheilSenRegressor()),
    ("Huber Regression", HuberRegressor()),
    ("Kernel Ridge", KernelRidge()),
    ("Gaussian Process Regression", GaussianProcessRegressor()),
    ("SVR", SVR())

]
