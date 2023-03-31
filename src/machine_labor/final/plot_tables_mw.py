import pandas as pd
import numpy as np
import plotly.graph_objects as go

def plot_precision_recall_curve(data):
    """Plots precision-recall curves for the gradient-boosting tree, random forest, decision tree, linear regression, and basic logistic models.
    Args:
        data (pd.DataFrame): A DataFrame containing the recall and precision values for each model to be plotted.

    Returns:
        fig : A plotly figure object representing plotted precision-recall curves for the gradient-boosting tree, random forest, decision tree, linear regression, and basic logistic models.

    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=data['recall_boost'], y=data['precision_boost'], mode='lines', name='Boosted Trees'))
    fig.add_trace(go.Scatter(x=data['recall_boost'], y=data['precision_interp_rf'], mode='lines', name='Random Forest'))
    fig.add_trace(go.Scatter(x=data['recall_boost'], y=data['precision_interp_tree'], mode='lines', name='Tree'))
    fig.add_trace(go.Scatter(x=data['recall_boost'], y=data['precision_interp_lm'], mode='lines', name='Linear (Card & Krueger)'))
    fig.add_trace(go.Scatter(x=data['recall_boost'], y=data['precision_interp_logit'], mode='lines', name='Basic Logistic'))

    fig.update_layout(title='Precision-Recall Curve',
                      xaxis_title='Recall',
                      yaxis_title='Precision')
    return fig

def plot_precision_relative_boost(data):
    """Plots the precision-recall curves for the random forest, decision tree, linear regression, and basic logistic models,
    relative to the gradient-boosting tree model.

    Args:
        data (pd.DataFrame): A DataFrame containing the recall and precision values for each model to be plotted.

    Returns:
        fig : A plotly figure object representing precision-recall curves of different models relative to the Boosting Tree model.

    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['recall_boost'], y=data['precision_diff_rf'], mode='lines', name='Random Forest'))
    fig.add_trace(go.Scatter(x=data['recall_boost'], y=data['precision_diff_tree'], mode='lines', name='Tree'))
    fig.add_trace(go.Scatter(x=data['recall_boost'], y=data['precision_diff_lm'], mode='lines', name='Linear (Card & Krueger)'))
    fig.add_trace(go.Scatter(x=data['recall_boost'], y=data['precision_diff_logit'], mode='lines', name='Basic Logistic'))
    fig.update_layout(title='Precision-Recall Curve',
                      xaxis_title='Recall',
                      yaxis_title='Precision relative to the Boosting')
    return fig

def feature_importance(boost_basic_model):
    """Plots the feature importance i.e. relative inﬂuences of the predictors in the gradient-boosting tree prediction model.

    Args:
        boost_income (sklearn.ensemble.GradientBoostingClassifier): A fitted boosting model.

    Returns:
        fig : A plotly figure displaying in a descending order the feature importance of each predictor in the gradient-boosting tree model.

    """
    feature_labels = ["age","race","sex","hispanic","dmarried","ruralstatus","educcat","veteran"]
    feature_importance = boost_basic_model.feature_importances_
    data = {"feature": feature_labels, "importance": feature_importance}
    data = pd.DataFrame(data)
    ind = np.argsort(data["importance"], axis=1)
    sorted_labels = data["importance"][ind]
    sorted_importance = data["feature"][ind]
    fig = go.Figure(data=[go.Bar(x=sorted_labels, y=sorted_importance, orientation='h')])
    fig.update_layout(title='Feature Importance',
                      yaxis_title='Feature',
                      xaxis_title='Importance')
    return fig


