import plotly.graph_objects as go


def style_chart(fig: go.Figure, y_title: str, y_prefix: str = "$") -> go.Figure:
    """Apply consistent professional styling to a Plotly line chart.

    Args:
        fig: A Plotly figure, typically from px.line().
        y_title: Label for the y-axis.
        y_prefix: Prefix shown before y-axis values and hover labels
            (e.g. "$" for currency, "" for plain numbers).

    Returns:
        The same figure, styled in place and returned for chaining.
    """
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Roman", size=13, color="#2C4F81"),
        title=dict(font=dict(size=20, color="#1e293b")),
        width=650,
        height=450,
        margin=dict(l=60, r=40, t=80, b=60),
    )
    fig.update_traces(
        line=dict(width=3, color="#08565f"),
        hovertemplate="%{x|%b %Y}<br>" + y_prefix + "%{y:.2f}<extra></extra>",
    )
    fig.update_xaxes(showgrid=False, title="Month")
    fig.update_yaxes(showgrid=True, gridcolor="#e2e8f0", title=y_title, tickprefix=y_prefix)
    return fig
