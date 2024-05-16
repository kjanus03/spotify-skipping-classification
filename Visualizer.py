import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter
from sklearn.metrics import ConfusionMatrixDisplay
import warnings
import os

warnings.filterwarnings("ignore", category=RuntimeWarning)


class Visualizer:
    def __init__(self, output_directory: str, palette: str):
        self.save_path = output_directory
        if not os.path.exists(output_directory):
            os.mkdir(output_directory)
        self.palette = palette
        self.legend_labels = ['Niepominięta', 'Pominięta']

    def skipped_countplot(self, df: pd.DataFrame) -> None:
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.countplot(data=df, x='skipped', hue='skipped', palette=self.palette, ax=ax)
        plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:,.0f}'))
        plt.title('Liczba pominiętych i niepominiętych piosenek', fontsize=18)
        plt.xlabel('Pominięta Piosenka', fontsize=15)
        plt.ylabel('Liczba Piosenek', fontsize=15)
        plt.legend(labels=self.legend_labels, loc='upper left', fontsize=15)
        plt.xticks(ticks=[0, 1], labels=['Niepominięta', 'Pominięta'], fontsize=15)
        plt.tight_layout()
        plt.savefig(f'{self.save_path}/skipped_countplot.png')
        plt.show()

    def platform_histogram(self, df: pd.DataFrame) -> None:
        platforms = ['iPhone', 'PC', 'Android', 'Other']
        df['platform'] = pd.Categorical(df['platform'], categories=platforms, ordered=True)

        fig, ax = plt.subplots(figsize=(12, 8))
        sns.histplot(
            data=df,
            x='platform',
            hue='skipped',
            palette=self.palette,
            ax=ax,
            multiple='stack',
            element='bars',
            log_scale=(False, True)
        )
        plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:,.0f}'))
        plt.title('Rozkład słuchanych piosenek według platformy', fontsize=18)
        plt.xlabel('Platforma', fontsize=15)
        plt.ylabel('Liczba piosenek', fontsize=15)
        plt.xticks(rotation=45, fontsize=15)
        plt.tight_layout()
        plt.savefig(f'{self.save_path}/platform_histogram.png')
        plt.show()

    def reason_start_histogram(self, df: pd.DataFrame) -> None:
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.histplot(data=df, x='reason_start', hue='skipped', palette=self.palette, ax=ax, multiple='stack',
                     element='bars')
        plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:,.0f}'))
        plt.title('Rozkład słuchanych piosenek według powodu rozpoczęcia', fontsize=18)
        plt.legend(labels=self.legend_labels, loc='upper right', fontsize=15)
        plt.xlabel('Powód Rozpoczęcia', fontsize=15)
        plt.ylabel('Liczba piosenek', fontsize=15)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{self.save_path}/reason_start_histogram.png')
        plt.show()

    def ms_played_histogram(self, df: pd.DataFrame) -> None:
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.histplot(data=df, x='ms_played', hue='skipped', palette=self.palette, ax=ax, log_scale=(True, False),
                     bins=50, multiple='stack', element='bars')
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:,.0f}'))
        plt.title('Rozkład czasu odsłuchu piosenek', fontsize=18)
        plt.xlabel('Czas odsłuchu [ms] (skala logarytmiczna)', fontsize=15)
        plt.ylabel('Liczba piosenek', fontsize=15)
        plt.legend(labels=self.legend_labels, loc='upper right', fontsize=15)
        plt.tight_layout()
        plt.savefig(f'{self.save_path}/ms_played_histogram.png')
        plt.show()

    def artist_skip_rate_stacked_barplot(self, df: pd.DataFrame) -> None:
        df_pivot = df.pivot_table(index='artist_skip_rate_bins', columns='skipped', aggfunc='size', fill_value=0)
        df_pivot_norm = df_pivot.div(df_pivot.sum(axis=1), axis=0)

        df_pivot_norm.plot(kind='bar', stacked=True, colormap=self.palette, figsize=(12, 8))
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        plt.title('Procentowy rozkład pomijania piosenek względem pomijania artystów', fontsize=18)
        plt.xlabel('Zakres Pomijania Artysty', fontsize=15)
        plt.ylabel('Procent Pominięć', fontsize=15)
        plt.legend(labels=self.legend_labels, loc='upper right', fontsize=15)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{self.save_path}/artist_skip_rate_stacked_barplot.png')
        plt.show()

    def hours_circular_bar_plot(self, df: pd.DataFrame) -> None:
        sns.set_style('whitegrid')
        x_max = 2 * np.pi
        df['angular_pos'] = np.linspace(0, x_max, 24, endpoint=False)

        cmap = plt.cm.get_cmap(self.palette, 24)
        colors = [cmap(i) for i in range(24)]

        df['colors'] = df['hour'].apply(lambda x: colors[x])
        fig, ax = plt.subplots(figsize=(12, 12), subplot_kw={'projection': 'polar'})

        ax.bar(df['angular_pos'], df['count'], alpha=1, color=df['colors'], linewidth=0, width=0.25, zorder=3)

        max_value = df['count'].max() + 1600
        r_offset = -6500
        ax.set_rlim(0, max_value)
        ax.set_rorigin(r_offset)

        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_xticks(ticks=df['angular_pos'], labels=[f'{i}:00' if i % 6 == 0 else '' for i in range(24)])
        ax.tick_params(axis='x', direction='in', pad=-272, labelsize=17, colors='black', labelcolor='black')
        ax.set_yticks(ticks=[])

        plt.title('Liczba odsłuchanych piosenek na Spotify w poszczególnych godzinach', fontsize=26, pad=20)
        plt.figtext(0.5, 0.05, 'Data: Spotify Extended Streaming History\nVisualization: Kacper Janus',
                    fontsize=12, ha='center', alpha=0.75)

        plt.savefig(f'{self.save_path}/hours_circular_bar_plot.png')
        plt.show()

    def times_of_day_barplot(self, df: pd.DataFrame, times_descriptions: dict[str, str]) -> None:
        sns.set_style('whitegrid')
        fig, ax = plt.subplots(figsize=(12, 10))
        barplot = sns.barplot(data=df, x='time_of_day', y='count', palette=self.palette, ax=ax, hue='time_of_day')

        legend_handles = []
        available_times = df['time_of_day'].unique().tolist()

        for time, desc in times_descriptions.items():
            if time in available_times:
                index = available_times.index(time)
                if index < len(barplot.patches):
                    color = barplot.patches[index].get_facecolor()
                    patch = mpatches.Patch(color=color, label=f'{time} {desc}')
                    legend_handles.append(patch)

        ax.legend(handles=legend_handles, loc='upper left', fontsize=12, title='Time of Day', title_fontsize='14')

        ax.set_xticks(range(len(available_times)))
        ax.set_xticklabels(available_times, ha='center', fontsize=12)

        ax.set_xlabel('Pora dnia', fontsize=15)
        ax.set_ylabel('Liczba piosenek', fontsize=15)
        ax.set_title('Liczba słuchanych piosenek według pory dnia', fontsize=18)

        plt.savefig(f'{self.save_path}/times_of_day_barplot.png')
        plt.show()

    def model_confusion_matrix(self, y_test: np.ndarray, y_pred: np.ndarray, model_name, sampling_desc="Brak") -> None:
        plt.figure(figsize=(16, 16))
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
        plt.suptitle(f'Macierz pomyłek', fontsize=22)
        plt.title(f'Model: {model_name}\nSampling: {sampling_desc}', fontsize=12, y=1.02)
        plt.tight_layout()
        plt.savefig(f'{self.save_path}/model_confusion_matrix_{model_name}_{sampling_desc}.png')
        plt.show()

    def correlations_heatmap(self, df: pd.DataFrame, features: list[str]) -> None:
        sns.set(style='white')
        corr = df[features].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        f, ax = plt.subplots(figsize=(11, 9))
        cmap = plt.cm.get_cmap(self.palette, 24).reversed()
        sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, square=True, linewidths=.5, annot_kws={"size": 10},
                    cbar_kws={"shrink": .75}, annot=True)
        plt.title('Macierz korelacji', fontsize=18)
        plt.tight_layout()
        plt.savefig(f'{self.save_path}/correlations_heatmap.png')
        plt.show()

    def platform_violinplot(self, df: pd.DataFrame) -> None:
        sns.set(style='whitegrid')
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.violinplot(x='platform', y='hour', hue='skipped', data=df, ax=ax, palette=self.palette)
        plt.xticks(fontsize=15)
        ax.set_xlabel('Platforma', fontsize=17)
        ax.set_ylabel('', fontsize=17)
        plt.legend(labels=['Niepominięta', 'Pominięta'], loc='upper right', fontsize=15)
        ax.set_title('Rozkład pomijania piosenek na Spotify według platformy i godziny', fontsize=18)
        plt.savefig(f'{self.save_path}/platform_violinplot.png')
        plt.show()
