import numpy as np
import plotly.graph_objects as go

def plot_precision_recall_curve(data):
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=data['recall_boost'], y=data['precision_boost'], mode='lines', name='Boosted Trees'))
    fig.add_trace(go.Scatter(x=data['recall_boost'], y=data['precision_interp_rf'], mode='lines', name='Random Forest'))
    fig.add_trace(go.Scatter(x=data['recall_boost'], y=data['precision_interp_tree'], mode='lines', name='Tree'))
    fig.add_trace(go.Scatter(x=data['recall_boost'], y=data['precision_interp_lm'], mode='lines', name='Linear (Card & Krueger)'))
    fig.add_trace(go.Scatter(x=data['recall_boost'], y=data['precision_interp_glm'], mode='lines', name='Basic Logistic'))

    fig.update_layout(title='Precision-Recall Curve',
                      xaxis_title='Recall',
                      yaxis_title='Precision')
    return fig

def plot_precision_relative_boost(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['recall_boost'], y=data['precision_diff_rf'], mode='lines', name='Random Forest'))
    fig.add_trace(go.Scatter(x=data['recall_boost'], y=data['precision_diff_tree'], mode='lines', name='Tree'))
    fig.add_trace(go.Scatter(x=data['recall_boost'], y=data['precision_diff_lm'], mode='lines', name='Linear (Card & Krueger)'))
    fig.add_trace(go.Scatter(x=data['recall_boost'], y=data['precision_diff_glm'], mode='lines', name='Basic Logistic'))
    fig.update_layout(title='Precision-Recall Curve',
                      xaxis_title='Recall',
                      yaxis_title='Precision relative to the Boosting')
    return fig

def feature_importance(boost_income):
    feature_importance = boost_income.feature_importances_
    sorted_idx = np.argsort(feature_importance)
    labels = [f'Feature {i+1}' for i in sorted_idx]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=feature_importance[sorted_idx],y=labels,orientation='h'))
    fig.update_layout(title='Feature Importance',
                      yaxis_title='Feature',
                      xaxis_title='Importance')
    return fig