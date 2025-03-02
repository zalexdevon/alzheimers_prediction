import pandas as pd
import os
from classifier import logger
from classifier.entity.config_entity import MonitorPlotterConfig
from classifier.Mylib import myfuncs
from sklearn import metrics
import plotly.express as px


class MonitorPlotter:
    def __init__(self, config: MonitorPlotterConfig):
        self.config = config

    def plot(self, monitor):

        monitor_descs = [item[0] for item in monitor]
        train_scores = [item[1] for item in monitor]
        val_scores = [item[2] for item in monitor]

        for i in range(len(train_scores)):
            if train_scores[i] > self.config.max_val_value:
                train_scores[i] = self.config.max_val_value

            if val_scores[i] > self.config.max_val_value:
                val_scores[i] = self.config.max_val_value

        x_values = list(range(1, len(train_scores) + 1))

        df = pd.DataFrame({"x": x_values, "train": train_scores, "val": val_scores})
        df_long = df.melt(
            id_vars=["x"],
            value_vars=["train", "val"],
            var_name="Category",
            value_name="y",
        )

        fig = px.line(
            df_long,
            x="x",
            y="y",
            color="Category",
            markers=True,
            color_discrete_map={
                "train": "gray",
                "val": "blue",
            },
            hover_data={"x": False, "Category": False},
        )

        for i in range(len(x_values)):
            text = monitor_descs[i]

            fig.add_annotation(
                text=text,
                x=x_values[i],
                y=self.config.max_val_value,
                xref="x",
                yref="y",
                showarrow=False,
                font=dict(family="Consolas", size=7, color="red"),
                ax=0,
                align="left",
                yanchor="bottom",
            )

        fig.add_hline(
            y=self.config.max_val_value,
            line_dash="solid",
            line_color="black",
            line_width=2,
        )

        fig.add_hline(
            y=self.config.target_val_value,
            line_dash="dash",
            line_color="green",
            line_width=2,
        )

        fig.update_layout(
            autosize=False,
            width=100 * (len(x_values) + 2) + 30,
            height=600,
            margin=dict(l=30, r=10, t=10, b=0),
            xaxis=dict(
                title="",
                range=[
                    0,
                    len(x_values) + 2,
                ],
                tickmode="linear",
            ),
            yaxis=dict(
                title="",
                range=[0, self.config.max_val_value + self.config.height_for_annot],
                dtick=self.config.dtick_y_value,
            ),
            showlegend=False,
        )

        html_path = "monitor_plot.html"
        fig.write_html(html_path, config={"displayModeBar": False})
