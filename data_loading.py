import glob
import pandas as pd
from SunTimes import SunTimes


def read_json_files(file_paths: glob.glob) -> pd.DataFrame:
    data_frames = [pd.read_json(file_path) for file_path in file_paths]
    return pd.concat(data_frames, ignore_index=True)


def load_data(sun_times: SunTimes, from_csv: bool = True, file_paths: glob.glob = None) -> pd.DataFrame:
    if file_paths is None:
        file_paths = glob.glob('./SpotifyExtendedStreamingHistory/Streaming_History_Audio_*.json')
    if from_csv:
        return pd.read_csv('Data/streaming_history.csv',
                           dtype={
                                    'ts': str,
                                    'username': str,
                                    'platform': str,
                                    'ms_played': int,
                                    'conn_country': str,
                                    'conn_device_type': str,
                                    'ip_addr_decrypted': str,
                                    'user_agent_decrypted': str,
                                    'master_metadata_track_name': str,
                                    'master_metadata_album_artist_name': str,
                                    'master_metadata_album_album_name': str,
                                    'spotify_track_uri': str,
                                    'episode_name': str,
                                    'episode_show_name': str,
                                    'spotify_episode_uri': str,
                                    'reason_start': str,
                                    'reason_end': str,
                                    'shuffle': bool,
                                    'offline': bool,
                                    'offline_timestamp': str,
                                    'incognito_mode': bool,
                                    'hour': int,
                                    'time_of_day': str
                                    }
                           )
    streaming_history = read_json_files(file_paths)
    streaming_history['ts'] = pd.to_datetime(streaming_history['ts'])
    streaming_history['hour'] = streaming_history['ts'].dt.hour
    streaming_history['time_of_day'] = streaming_history['ts'].apply(sun_times.get_time_of_day)
    return streaming_history


def save_data(df: pd.DataFrame, file_path: str) -> None:
    df.to_csv('Data/streaming_history.csv', index=False)
