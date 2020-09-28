import numpy as np

def get_metrics(results):
    """Get the Residual Standard Error, R Squared and F Statistic
    from statsmodels OLS result

    Args:
        results (RegressionResultsWrapper):
            statsmodels linear regression fit summary

    Returns:
        rse (float): Residual Standard Error / Root Mean Squared Error
        r_squared (float): R Squared
        f_statistic (float): F statistic
    """
    
    rse = np.sqrt(results.mse_resid).round(2)
    r_squared = results.rsquared.round(2)
    f_statistic = results.fvalue.round(2)
    return rse, r_squared, f_statistic

def display_metrics(rse, r_squared, f_statistic):
    """Print the Residual Standard Error, R Squared and F Statistic

    Args:
        rse (float): Residual Standard Error / Root Mean Squared Error
        r_squared (float): R Squared
        f_statistic (float): F statistic

    Returns:
        None
    """
    
    print(f'Residual Standard Error: {rse}')
    print(f'R Squared: {r_squared}')
    print(f'F Statistic: {f_statistic}\n')
