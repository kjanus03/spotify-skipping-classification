import glob
import numpy as np
import pandas as pd
from data_loading import load_data, save_data
from data_preprocessing import preprocess_data, process_platform, process_skipped, process_shuffle, \
    undersample_preprocessed_data, oversample_preprocessed_data
from data_analysis import analyze_data_with_dtc
from utils import print_results, create_report
from Visualizer import Visualizer
from SunTimes import SunTimes


def load_and_save_data(sun_times: SunTimes, from_csv: bool, file_paths: glob.glob):
    streaming_history = load_data(sun_times, from_csv=from_csv, file_paths=file_paths)
    save_data(streaming_history, 'Data/streaming_history.csv')
    return streaming_history


def visualize_initial_data(visualizer: Visualizer, streaming_history: pd.DataFrame) -> pd.DataFrame:
    hours_series = streaming_history.groupby('hour').size().reset_index(name='count')
    visualizer.hours_circular_bar_plot(hours_series)
    streaming_history = process_skipped(streaming_history)
    visualizer.skipped_countplot(streaming_history)
    visualizer.ms_played_histogram(streaming_history)
    visualizer.reason_start_histogram(streaming_history)
    return streaming_history


def process_and_visualize_data(visualizer: Visualizer, streaming_history: pd.DataFrame) -> pd.DataFrame:
    streaming_history = process_platform(streaming_history)
    streaming_history = process_shuffle(streaming_history)

    streaming_history['play_count'] = streaming_history.groupby('spotify_track_uri').cumcount()
    streaming_history['skipped_last_time'] = streaming_history.groupby('spotify_track_uri')['skipped'].shift().fillna(
        False).astype(int)
    streaming_history['album_play_count'] = streaming_history.groupby('master_metadata_album_album_name').cumcount()
    streaming_history['artist_play_count'] = streaming_history.groupby('master_metadata_album_artist_name').cumcount()
    streaming_history['artist_skip_rate_so_far'] = streaming_history.groupby('master_metadata_album_artist_name')[
                                                       'skipped'].cumsum() / (
                                                               streaming_history['artist_play_count'] + 1)
    streaming_history['previous_skipped'] = streaming_history['skipped'].shift().fillna(False).astype(int)

    shifted_skipped = streaming_history['skipped'].shift(fill_value=0)
    not_skipped_diff = shifted_skipped != 0
    streak_group = not_skipped_diff.cumsum()
    streaming_history['current_listening_streak'] = streaming_history.groupby(streak_group).cumcount().where(
        shifted_skipped == 0, 0)

    bins = np.linspace(0, 1, 21)
    streaming_history['artist_skip_rate_bins'] = pd.cut(streaming_history['artist_skip_rate_so_far'], bins=bins,
                                                        labels=[f"{b * 100:.1f}-{b_next * 100:.1f}%" for b, b_next in
                                                                zip(bins[:-1], bins[1:])])

    visualizer.platform_histogram(streaming_history)
    visualizer.platform_violinplot(streaming_history)
    visualizer.artist_skip_rate_stacked_barplot(streaming_history)

    return streaming_history


def visualize_time_of_day_data(visualizer: Visualizer, streaming_history: pd.DataFrame, sun_times: SunTimes) -> None:
    times_order = ['wschód słońca', 'rano', 'środek dnia', 'po południu', 'zachód słońca', 'noc']
    times_df = streaming_history.groupby('time_of_day').size().reindex(times_order).reset_index(name='count')
    visualizer.times_of_day_barplot(times_df, times_descriptions=sun_times.times_descriptions)


def create_and_visualize_correlations(visualizer: Visualizer, streaming_history: pd.DataFrame, numerical_features: list[str]) -> None:
    visualizer.correlations_heatmap(streaming_history, numerical_features + ['skipped'])


def run_analysis_and_print_results(streaming_history:pd.DataFrame, categorical_features: list[str], numerical_features: list[str], visualizer: Visualizer) -> None:
    X_train, X_test, y_train, y_test = preprocess_data(streaming_history, categorical_features, numerical_features)
    X_train_undersampled, y_train_undersampled = undersample_preprocessed_data(X_train, y_train)
    X_train_oversampled, y_train_oversampled = oversample_preprocessed_data(X_train, y_train)

    y_pred = analyze_data_with_dtc(X_train, X_test, y_train)
    y_pred_undersmpled = analyze_data_with_dtc(X_train_undersampled, X_test, y_train_undersampled)
    y_pred_oversampled = analyze_data_with_dtc(X_train_oversampled, X_test, y_train_oversampled)

    print('Bez undersamplingu:')
    print_results(y_test, y_pred)
    print('Z undersamplingiem:')
    print_results(y_test, y_pred_undersmpled)
    print('Z oversamplingiem:')
    print_results(y_test, y_pred_oversampled)

    visualizer.model_confusion_matrix(y_test, y_pred, model_name='Drzewo decyzyjne')
    visualizer.model_confusion_matrix(y_test, y_pred_undersmpled, model_name='Drzewo decyzyjne',
                                      sampling_desc='Zaniżanie próbkowania danych (undersampling)')
    visualizer.model_confusion_matrix(y_test, y_pred_oversampled, model_name='Drzewo decyzyjne',
                                      sampling_desc='Zawyżanie próbkowania danych (oversampling)')


def main():
    file_paths = glob.glob('./SpotifyExtendedStreamingHistory/Streaming_History_Audio_*.json')
    sun_times = SunTimes()
    visualizer = Visualizer('Output', 'viridis')

    streaming_history = load_and_save_data(sun_times=sun_times, from_csv=False, file_paths=file_paths)
    streaming_history = visualize_initial_data(visualizer=visualizer, streaming_history=streaming_history)
    streaming_history = process_and_visualize_data(visualizer=visualizer, streaming_history=streaming_history)
    visualize_time_of_day_data(visualizer=visualizer, streaming_history=streaming_history, sun_times=sun_times)

    categorical_features = ['ts', 'time_of_day', 'master_metadata_album_artist_name',
                            'master_metadata_album_album_name', 'reason_start', 'platform']
    numerical_features = ['ms_played', 'play_count', 'artist_play_count', 'album_play_count', 'shuffle',
                          'artist_skip_rate_so_far', 'previous_skipped', 'skipped_last_time', 'hour']

    create_report(df=streaming_history, output_directory='Output/')
    create_and_visualize_correlations(visualizer, streaming_history, numerical_features)

    run_analysis_and_print_results(streaming_history, categorical_features, numerical_features, visualizer)


if __name__ == '__main__':
    main()
