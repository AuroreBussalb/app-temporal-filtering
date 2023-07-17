#!/usr/local/bin/python3

import json
import mne
import warnings
import numpy as np
import os
import shutil
import pandas as pd


def convert_parameters_to_None(config):
    """Convert parameters whose value is "" into None.
    Parameters
    ----------
    config: dict
        Dictionary containing all the parameters of the App.
    Returns
    -------
    config: dict
        Dictionary with parameters converted to None where needed.
    """

    # Convert all "" to None when the App runs on BL
    tmp = dict((k, None) for k, v in config.items() if v == "")
    config.update(tmp)

    return config


def read_optional_files(config, out_dir_name):
    """Read all optional files given to the App.
    Parameters
    ----------
    config: dict
        Dictionary containing all the parameters of the App.
    out_dir_name: str
        Name of the output directory of the App.
    Returns
    -------
    config: dict
        Dictionary with parameters minus the optional files entries.
    cross_talk_file: str or None
        Path to the FIF file with cross-talk correction information.
    calibration_file: str or None
        Path to the '.dat' file with fine calibration coefficients. This file is machine/site-specific.
    events_file: str or None
        Path to the '.tsv' BIDS compliant file containing events.
    head_pos_file: str or None
        Path to the '.pos' file containing the info to perform movement compensation.
    channels_file: str or None
        Path to the '.tsv' file containing channel info.
    destination: str or None
        Path to the FIF file containing the destination location for the head.
    """

    # From meg/fif datatype #

    # Read the crosstalk file
    if 'crosstalk' in config.keys():  # if the App has no meg/fif input this key doesn't exist
        cross_talk_file = config.pop('crosstalk')
        if cross_talk_file is not None:  # when the App runs locally the value is None when no file is given
            if os.path.exists(
                    cross_talk_file) is False:  # on BL a path is always created even when the file doesn't exist
                cross_talk_file = None
            else:
                shutil.copy2(cross_talk_file,
                             os.path.join(out_dir_name, 'crosstalk_meg.fif'))  # required to run a pipeline on BL
    else:
        cross_talk_file = None  # we need it later in this function and also in function message_optional_files_in_reports

    # Read the calibration file
    if 'calibration' in config.keys():
        calibration_file = config.pop('calibration')
        if calibration_file is not None:
            if os.path.exists(calibration_file) is False:
                calibration_file = None
            else:
                shutil.copy2(calibration_file, os.path.join(out_dir_name, 'calibration_meg.dat'))
    else:
        calibration_file = None

    # Read the events file
    # We don't copy this file in outdir yet because this file can be given in fif-override
    # and we take the fif override file by default
    if 'events' in config.keys():
        events_file = config.pop('events')
        if events_file is not None:
            if os.path.exists(events_file) is False:
                events_file = None
    else:
        events_file = None

    # Read head pos file
    # We don't copy this file in outdir yet because this file can be given in fif-override
    # and we take the fif override file by default
    if 'headshape' in config.keys():
        head_pos_file = config.pop('headshape')
        if head_pos_file is not None:
            if os.path.exists(head_pos_file) is False:
                head_pos_file = None
    else:
        head_pos_file = None

    # Read channels file
    # We don't copy this file in outdir yet because this file can be given in fif-override
    # and we take the fif override file by default
    if 'channels' in config.keys():
        channels_file = config.pop('channels')
        if channels_file is not None:
            if os.path.exists(channels_file) is False:
                channels_file = None
    else:
        channels_file = None

        # Read destination file
    # We don't copy this file in outdir yet because this file can be given in fif-override
    # and we take the fif override file by default
    if 'destination' in config.keys():
        destination = config.pop('destination')
        if destination is not None:
            if os.path.exists(destination) is False:
                destination = None
    else:
        destination = None

    # From meg/fif-override datatype #

    # Read the destination file
    if 'destination_override' in config.keys():  # if the App has no meg/fif-override input this key doesn't exist
        destination_override = config.pop('destination_override')
        # No need to test if destination_override is None, this key is only present when the app runs on BL
        if os.path.exists(destination_override) is False:
            if destination is not None:
                # If destination from meg/fif exists but destination_override from meg/fif-override doesn't,
                # we copy it in out_dir
                shutil.copy2(destination,
                             os.path.join(out_dir_name, 'destination.fif'))  # required to run a pipeline on BL
        else:
            # If destination_override from meg/fif-override exists, we copy it in out_dir
            # By default we copy the files given in input of meg/fif-override
            shutil.copy2(destination_override,
                         os.path.join(out_dir_name, 'destination.fif'))  # required to run a pipeline on BL
            destination = destination_override  # we overwrite the value of destination
    else:
        # If the App has no meg/override datatype (or if the App is run locally)
        if destination is not None:
            # If destination file from meg/fif is not None, we copy it in outdir
            shutil.copy2(destination, os.path.join(out_dir_name, 'destination.fif'))

    # Read head pos file
    if 'headshape_override' in config.keys():
        head_pos_file_override = config.pop('headshape_override')
        # No need to test if headshape_override is None, this key is only present when the app runs on BL
        if os.path.exists(head_pos_file_override) is False:
            if head_pos_file is not None:
                shutil.copy2(head_pos_file, os.path.join(out_dir_name, 'headshape.pos'))
                head_pos_file = mne.chpi.read_head_pos(head_pos_file)
        else:
            shutil.copy2(head_pos_file_override, os.path.join(out_dir_name, 'headshape.pos'))
            head_pos_file = mne.chpi.read_head_pos(head_pos_file_override)
    else:
        if head_pos_file is not None:
            shutil.copy2(head_pos_file, os.path.join(out_dir_name, 'headshape.pos'))
            head_pos_file = mne.chpi.read_head_pos(head_pos_file)

    # Read channels file
    if 'channels_override' in config.keys():
        channels_file_override = config.pop('channels_override')
        # No need to test if channels_override is None, this key is only present when the app runs on BL
        if os.path.exists(channels_file_override) is False:
            if channels_file is not None:
                shutil.copy2(channels_file, os.path.join(out_dir_name, 'channels.tsv'))
        else:
            shutil.copy2(channels_file_override, os.path.join(out_dir_name, 'channels.tsv'))
            channels_file = channels_file_override
    else:
        if channels_file is not None:
            shutil.copy2(channels_file, os.path.join(out_dir_name, 'channels.tsv'))

            # Read the events file
    if "events_override" in config.keys():
        events_file_override = config.pop('events_override')
        # No need to test if events_override is None, this key is only present when the app runs on BL
        if os.path.exists(events_file_override) is False:
            if events_file is not None:
                shutil.copy2(events_file, os.path.join(out_dir_name, 'events.tsv'))
        else:
            shutil.copy2(events_file_override,
                         os.path.join(out_dir_name, 'events.tsv'))  # required to run a pipeline on BL
            events_file = events_file_override
    else:
        if events_file is not None:
            shutil.copy2(events_file, os.path.join(out_dir_name, 'events.tsv'))

    return config, cross_talk_file, calibration_file, events_file, head_pos_file, channels_file, destination


def update_data_info_bads(data, channels_file):
    """Update data.info['bads'] with the info contained in channels.tsv.
    Parameters
    ----------
    data: instance of mne.io.Raw or instance of mne.Epochs
        Data whose info['bads'] needs to be updated.
    channels_file: str
        BIDS compliant channels.tsv corresponding to data.
    Returns
    -------
    data: instance of mne.io.Raw or instance of mne.Epochs
        Data whose info['bads'] has been updated.
    user_warning_message_channels: str
        Message to be displayed on BL UI if data.info['bads'] is updated.
    """

    # Convert channels.tsv into a dataframe
    df_channels = pd.read_csv(channels_file, sep='\t')

    # Select bad channels' name
    bad_channels = df_channels[df_channels["status"] == "bad"]['name']
    bad_channels = list(bad_channels.values)

    # Populate data.info['bads'] with channels.tsv info #

    # Sort them in order to compare them
    data.info['bads'].sort()
    bad_channels.sort()

    # Warning message if they are different
    if data.info['bads'] != bad_channels:
        user_warning_message_channels = f'Bad channels from the info of your MEG file are different from ' \
                                        f'those in the channels.tsv file. By default, only bad channels from channels.tsv ' \
                                        f'are considered as bad: the info of your MEG file is updated with those channels.'
        # Add bad_channels to data.info['bads']
        data.info['bads'] = bad_channels
    else:
        user_warning_message_channels = None

    return data, user_warning_message_channels


def message_optional_files_in_reports(calibration_file, cross_talk_file, head_pos_file, destination):
    """Create messages regarding the presence of the optional files, which will be
    later added in html reports of Apps.
    Parameters
    ----------
    calibration_file: str or None
        Path to the '.dat' file with fine calibration coefficients. This file is machine/site-specific.
    cross_talk_file: str or None
        Path to the FIF file with cross-talk correction information.
    head_pos_file: str or None
        '.pos' file containing the info to perform movement compensation.
    destination: str or None
        The destination location for the head.
    Returns
    -------
    report_calibration_file: str
        message regarding the calibration file.
    report_cross_talk_file: str
        message regarding the cross-talk file.
    report_head_pos_file: str
        message regarding the head pos file.
    report_destination_file: str
        message regarding the destination file.
    """

    # Calibration file
    if calibration_file is None:
        report_calibration_file = 'No calibration file provided'
    else:
        report_calibration_file = 'Calibration file provided'

        # Cross talk file
    if cross_talk_file is None:
        report_cross_talk_file = 'No cross-talk file provided'
    else:
        report_cross_talk_file = 'Cross-talk file provided'

        # Head pos file
    if head_pos_file is None:
        report_head_pos_file = 'No headshape file provided'
    else:
        report_head_pos_file = 'Headshape file provided'

    # Destination
    if destination is None:
        report_destination_file = 'No destination file provided'
    else:
        report_destination_file = 'Destination file provided'

    return report_calibration_file, report_cross_talk_file, report_head_pos_file, report_destination_file


def define_kwargs(config):
    """Define kwargs for the mne functions used by the App
    Parameters
    ----------
    config: dict
        Dictionary containing all the parameters of the App.
    Returns
    -------
    config: dict
        Dictionary containing all the parameters to apply the mne function.
    """

    # Delete keys values in config.json when the App is executed on Brainlife
    if '_app' and '_tid' and '_inputs' and '_outputs' in config.keys():
        del config['_app'], config['_tid'], config['_inputs'], config['_outputs']

        # When you run a pipeline rule, another key appeers
    if "_rule" in config.keys():
        del config['_rule']

    return config