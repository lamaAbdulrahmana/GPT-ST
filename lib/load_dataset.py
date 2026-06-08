import os
import numpy as np
from datetime import datetime, timedelta
from lib.event_utils import build_event_dataset, split_events_chronological

def time_add(data, week_start, interval=5, weekday_only=False, holiday_list=None,
             day_start=0, hour_of_day=24, start_date=None, holiday_dates=None):
    if weekday_only:
        week_max = 5
    else:
        week_max = 7
    time_slot = hour_of_day * 60 // interval
    day_data = np.zeros_like(data)
    week_data = np.zeros_like(data)
    holiday_data = np.zeros_like(data)
    day_init = day_start
    week_init = week_start
    holiday_init = 1
    for index in range(data.shape[0]):
        if (index) % time_slot == 0:
            day_init = day_start
        day_init = day_init + 1
        if (index) % time_slot == 0 and index != 0:
            week_init = week_init + 1
        if week_init > week_max:
            week_init = 1
        if week_init < 6:
            holiday_init = 1
        else:
            holiday_init = 2

        day_data[index:index + 1, :] = day_init
        week_data[index:index + 1, :] = week_init
        holiday_data[index:index + 1, :] = holiday_init

    if holiday_list is not None:
        for j in holiday_list:
            holiday_data[(j - 1) * time_slot:j * time_slot, :] = 2

    if start_date is not None and holiday_dates is not None:
        total_days = data.shape[0] // time_slot + 1
        for day_offset in range(total_days):
            current_date = start_date + timedelta(days=day_offset)
            if (current_date.month, current_date.day) in holiday_dates:
                ts_start = day_offset * time_slot
                ts_end = (day_offset + 1) * time_slot
                holiday_data[ts_start:ts_end, :] = 2

    return day_data, week_data, holiday_data


def load_st_dataset(dataset, args):
    if dataset == 'PEMS08':
        data_path = os.path.join('../data/PEMS08/PEMS08.npz')
        data = np.load(data_path)['data'][:, :, 0]  # only the first dimension, traffic flow data
        print(data.shape, data[data==0].shape)
        week_start = 5
        holiday_list = [4]
        interval = 5
        week_day = 7
        args.interval = interval
        args.week_day = week_day
        day_data, week_data, holiday_data = time_add(data, week_start, interval=interval, weekday_only=False, holiday_list=holiday_list)
    elif dataset == 'METR_LA':
        data_path = os.path.join('../data/METR_LA/metr_la.npz')
        data = np.load(data_path)['data']  # only traffic speed data
        print(data.shape, data[data == 0].shape)
        # print(sss)
        interval = 5
        week_day = 7
        args.interval = interval
        args.week_day = week_day
        week_start = 4
        holiday_list = [88]
        day_data, week_data, holiday_data = time_add(data, week_start, interval=interval, weekday_only=False,
                                                     holiday_list=holiday_list)
    elif dataset == 'NYC_BIKE':
        data_path = os.path.join('../data/NYC_BIKE/NYC_BIKE.npz')
        data = np.load(data_path)['data']  # DROP & PICK
        week_start = 5
        weekday_only = False
        interval = 30
        week_day = 7
        args.interval = interval
        args.week_day = week_day
        holiday_list = []
        day_data, week_data, holiday_data = time_add(data[..., 0], week_start, interval, weekday_only, holiday_list=holiday_list)
    elif dataset == 'NYC_TAXI':
        data_path = os.path.join('../data/NYC_TAXI/NYC_TAXI.npz')
        data = np.load(data_path)['data']  # DROP & PICK
        week_start = 5
        weekday_only = False
        interval = 30
        week_day = 7
        args.interval = interval
        args.week_day = week_day
        holiday_list = []
        day_data, week_data, holiday_data = time_add(data[..., 0], week_start, interval, weekday_only, holiday_list=holiday_list)
    elif dataset == 'RIYADH':
        data_path = os.path.join('../data/RIYADH/RIYADH.npz')
        data = np.load(data_path)['data']
        print(data.shape, data[data==0].shape)
        week_start = 2
        holiday_list = []
        interval = 10
        week_day = 7
        hour_of_day = 19
        args.interval = interval
        args.week_day = week_day
        args.hour_of_day = hour_of_day
        start_date = datetime(2025, 11, 4)
        holiday_dates = {(2, 22), (9, 23)}  # Founding Day, National Day
        day_data, week_data, holiday_data = time_add(
            data[..., 0] if data.ndim > 2 else data, week_start,
            interval=interval, weekday_only=False, holiday_list=holiday_list,
            start_date=start_date, holiday_dates=holiday_dates)
    else:
        raise ValueError
    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=-1)
        day_data = np.expand_dims(day_data, axis=-1).astype(int)
        week_data = np.expand_dims(week_data, axis=-1).astype(int)
        holiday_data = np.expand_dims(holiday_data, axis=-1).astype(int)
        data = np.concatenate([data, day_data, week_data, holiday_data], axis=-1)
    elif len(data.shape) > 2:
        day_data = np.expand_dims(day_data, axis=-1).astype(int)
        week_data = np.expand_dims(week_data, axis=-1).astype(int)
        holiday_data = np.expand_dims(holiday_data, axis=-1).astype(int)
        data = np.concatenate([data, day_data, week_data, holiday_data], axis=-1)
    else:
        raise ValueError
    print('Load %s Dataset shaped: ' % dataset, data.shape, data[..., 0:1].max(), data[..., 0:1].min(),
          data[..., 0:1].mean(), np.median(data[..., 0:1]), data.dtype)
    return data


def load_event_dataset(args):
    """
    Load RIYADH event segments for fine-tuning.
    Extracts windows from the FULL data (with time features already computed)
    so that day/week encodings are correct for each event's actual position.
    """
    full_data = load_st_dataset('RIYADH', args)
    print('Full RIYADH data with time features:', full_data.shape)

    events_path = os.path.join('../data/RIYADH/RIYADH_events.npz')
    start_date = datetime(2025, 11, 4)
    interval = 10
    hour_of_day = 19
    day_start_hour = 7

    from lib.event_utils import extract_event_windows, split_events_chronological
    segments, event_meta = extract_event_windows(
        full_data, events_path, start_date,
        interval=interval, hour_of_day=hour_of_day,
        day_start_hour=day_start_hour,
        pre_buffer=12, post_buffer=12
    )

    print(f'Extracted {len(event_meta)} event segments')
    for i, meta in enumerate(event_meta):
        print(f'  Event {i}: {meta["event_name"]} ({meta["event_type"]}) '
              f'steps [{meta["win_start"]}:{meta["win_end"]}] '
              f'len={meta["win_end"]-meta["win_start"]}')

    train_idx, val_idx, test_idx = split_events_chronological(event_meta)
    print(f'Split: {len(train_idx)} train, {len(val_idx)} val, {len(test_idx)} test events')

    train_segments = [segments[i] for i in train_idx]
    val_segments = [segments[i] for i in val_idx]
    test_segments = [segments[i] for i in test_idx]

    train_data = np.concatenate(train_segments, axis=0) if train_segments else np.array([])
    val_data = np.concatenate(val_segments, axis=0) if val_segments else np.array([])
    test_data = np.concatenate(test_segments, axis=0) if test_segments else np.array([])

    print(f'Event data shapes — train: {train_data.shape}, val: {val_data.shape}, test: {test_data.shape}')

    return train_data, val_data, test_data, event_meta, train_idx
