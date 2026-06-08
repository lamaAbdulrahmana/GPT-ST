import numpy as np
from datetime import datetime, timedelta


def datetime_to_timestep(dt, start_date, interval=10, hour_of_day=19, day_start_hour=7):
    """
    Map a datetime to a timestep index in the RIYADH dataset.

    The data has `hour_of_day` hours per day starting at `day_start_hour`,
    with `interval`-minute resolution. When the window crosses midnight
    (day_start_hour + hour_of_day > 24), early-morning hours (e.g. 0:00-2:00)
    belong to the previous calendar day's data window.
    """
    slots_per_day = hour_of_day * 60 // interval
    day_end_hour = day_start_hour + hour_of_day
    crosses_midnight = day_end_hour > 24
    overflow_hour = day_end_hour - 24 if crosses_midnight else 0

    day_offset = (dt.date() - start_date.date()).days
    hour_minute = dt.hour + dt.minute / 60.0

    if crosses_midnight and hour_minute < overflow_hour:
        day_offset -= 1
        hour_minute += 24

    if day_offset < 0:
        return 0

    if hour_minute < day_start_hour:
        slot_in_day = 0
    elif hour_minute >= day_end_hour:
        slot_in_day = slots_per_day - 1
    else:
        minutes_since_day_start = int((hour_minute - day_start_hour) * 60)
        slot_in_day = minutes_since_day_start // interval

    return day_offset * slots_per_day + slot_in_day


def extract_event_windows(full_data, events_npz_path, start_date,
                          interval=10, hour_of_day=19, day_start_hour=7,
                          pre_buffer=12, post_buffer=12):
    """
    Extract traffic data windows around each event.

    Args:
        full_data: np.ndarray (T, N, D) — full RIYADH traffic tensor
        events_npz_path: path to RIYADH_events.npz
        start_date: datetime — dataset start date
        interval: minutes per timestep
        hour_of_day: active hours per day in the dataset
        day_start_hour: hour when daily recording starts
        pre_buffer: timesteps to include before event start
        post_buffer: timesteps to include after event end

    Returns:
        segments: list of np.ndarray, each (window_len, N, D)
        event_meta: list of dicts with event metadata
    """
    events = np.load(events_npz_path, allow_pickle=True)
    event_names = events['event_name']
    start_dts = events['start_datetime']
    end_dts = events['end_datetime']
    event_types = events['event_type']
    time_verified = events['time_verified']

    total_timesteps = full_data.shape[0]
    segments = []
    event_meta = []

    for i in range(len(event_names)):
        evt_start = start_dts[i].astype('datetime64[s]').astype(datetime)
        evt_end = end_dts[i].astype('datetime64[s]').astype(datetime)

        ts_start = datetime_to_timestep(evt_start, start_date, interval,
                                        hour_of_day, day_start_hour)
        ts_end = datetime_to_timestep(evt_end, start_date, interval,
                                      hour_of_day, day_start_hour)

        win_start = max(0, ts_start - pre_buffer)
        win_end = min(total_timesteps, ts_end + post_buffer)

        if win_end <= win_start:
            continue

        segment = full_data[win_start:win_end]
        segments.append(segment)
        event_meta.append({
            'event_name': str(event_names[i]),
            'event_type': str(event_types[i]),
            'time_verified': bool(time_verified[i]),
            'ts_start': ts_start,
            'ts_end': ts_end,
            'win_start': win_start,
            'win_end': win_end,
            'original_start': evt_start,
            'original_end': evt_end,
        })

    return segments, event_meta


def build_event_dataset(full_data, events_npz_path, start_date,
                        interval=10, hour_of_day=19, day_start_hour=7,
                        pre_buffer=12, post_buffer=12):
    """
    Build a concatenated event dataset from event windows.
    Includes sliding windows within each event segment for data augmentation.

    Returns:
        event_data: np.ndarray (T_total, N, D) — concatenated event segments
        segment_boundaries: list of (start_idx, end_idx) in event_data
        event_meta: list of dicts
    """
    segments, event_meta = extract_event_windows(
        full_data, events_npz_path, start_date,
        interval, hour_of_day, day_start_hour,
        pre_buffer, post_buffer
    )

    if not segments:
        raise ValueError("No valid event segments extracted")

    segment_boundaries = []
    offset = 0
    for seg in segments:
        segment_boundaries.append((offset, offset + len(seg)))
        offset += len(seg)

    event_data = np.concatenate(segments, axis=0)
    return event_data, segment_boundaries, event_meta


def split_events_chronological(event_meta, train_ratio=0.7, val_ratio=0.15):
    """
    Split events chronologically for train/val/test.
    Events are sorted by start time; earliest go to train, latest to test.
    """
    n = len(event_meta)
    sorted_indices = sorted(range(n), key=lambda i: event_meta[i]['ts_start'])

    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio))

    train_idx = sorted_indices[:n_train]
    val_idx = sorted_indices[n_train:n_train + n_val]
    test_idx = sorted_indices[n_train + n_val:]

    if not test_idx:
        test_idx = val_idx[-1:]
        val_idx = val_idx[:-1]

    return train_idx, val_idx, test_idx
