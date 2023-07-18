#!/usr/local/bin/python3

import json
import mne
import numpy as np
import os
import pandas as pd
import warnings
from brainlife_apps_helper import helper


def temporal_filtering(data, param_epoched_data, param_l_freq, param_h_freq, param_picks_by_channel_types_or_names, 
                       param_filter_length, param_picks_by_channel_indices,
                       param_l_trans_bandwidth, param_h_trans_bandwidth, param_n_jobs,
                       param_method, param_iir_params, param_phase, param_fir_window,
                       param_fir_design, param_skip_by_annotation, param_raw_pad, param_epoch_pad):
    """Perform filtering using MNE Python and save the file once filtered.

    Parameters
    ----------
    data: instance of mne.io.Raw or instance of mne.Epochs
        Data to be filtered.
    param_epoched_data: bool
        If True, the data to be filtered is epoched, else it is continuous.
    param_l_freq: float or None
        For FIR filters, the lower pass-band edge; for IIR filters, the lower cutoff frequency. If None the 
        data are only low-passed.
    param_h_freq: float or None
        For FIR filters, the upper pass-band edge; for IIR filters, the upper cutoff frequency. If None the 
        data are only high-passed.
    param_picks_by_channel_types_or_names: str, list of str, or None 
        Channels to include. In lists, channel type strings (e.g., ["meg", "eeg"]) will pick channels of those types, channel name 
        strings (e.g., ["MEG0111", "MEG2623"]) will pick the given channels. Can also be the string values “all” 
        to pick all channels, or “data” to pick data channels. None (default) will pick all data channels. Note 
        that channels in info['bads'] will be included if their names are explicitly provided.
    param_picks_by_channel_indices: list of int, slice, or None
        Channels to include. Slices (e.g., "0, 10, 2" or "0, 10" if you don't want a step) and lists of integers 
        are interpreted as channel indices. None (default) will pick all data channels. This parameter must be set 
        to None if param_picks_by_channel_types_or_names is not None. Note that channels in info['bads'] will 
        be included if their indices are explicitly provided.
    param_filter_length: str or int
        Length of the FIR filter to use (if applicable). Can be ‘auto’ (default) : the filter length is chosen based 
        on the size of the transition regions, or an other str (human-readable time in units of “s” or “ms”: 
        e.g., “10s” or “5500ms”). If int, specified length in samples. For fir_design=”firwin”, this should not be used.
    param_l_trans_bandwidth: float or str
        Width of the transition band at the low cut-off frequency in Hz (high pass or cutoff 1 in bandpass). 
        Can be “auto” (default) to use a multiple of l_freq.     
    param_h_trans_bandwidth: float or str   
        Width of the transition band at the high cut-off frequency in Hz (low pass or cutoff 2 in bandpass). 
        Can be “auto” (default) to use a multiple of h_freq.
    param_n_jobs: int or str
        Number of jobs to run in parallel. Can be ‘cuda’ if cupy is installed properly and method=’fir’.
    param_method: str
        ‘fir’ will use overlap-add FIR filtering, ‘iir’ will use IIR forward-backward filtering (via filtfilt).
    param_iir_params: dict or None
        Dictionary of parameters to use for IIR filtering. If iir_params is None and method=”iir”, 
        4th order Butterworth will be used. 
    param_phase: str
        Phase of the filter, only used if method='fir'. Either 'zero' or 'zero-double'.
    param_fir_window: str
        The window to use in FIR design, can be “hamming” (default), “hann” (default in 0.13), or “blackman”.
    param_fir_design: str
        Can be “firwin” (default) or “firwin2”.
    param_skip_by_annotation: str or list of str
        If a string (or list of str), any annotation segment that begins with the given string will not be included in
        filtering, and segments on either side of the given excluded annotated segment will be filtered separately.
    param_raw_pad: str
        The type of padding to use for raw data. Supports all numpy.pad() mode options. Can also be 
        “reflect_limited” (default) and "edge".
    param_epoch_pad: str
        The type of padding to use for epoched data. Supports all numpy.pad() mode options. Can also be 
        “reflect_limited” and "edge" (default).

    Returns
    -------
    data_filtered: instance of mne.io.Raw or instance of mne.Epochs
        The filtered data.
    """

    # Raise error if both param picks are not None
    if param_picks_by_channel_types_or_names is not None and param_picks_by_channel_indices is not None:
        value_error_message = f"You can't provide values for both " \
                              f"param_picks_by_channel_types_or_names and " \
                              f"param_picks_by_channel_indices. One of them must be " \
                              f"set to None."
        raise ValueError(value_error_message)
    # Define param_picks
    elif param_picks_by_channel_types_or_names is None and param_picks_by_channel_indices is not None:
        param_picks = param_picks_by_channel_indices
    elif param_picks_by_channel_types_or_names is not None and param_picks_by_channel_indices is None:
        param_picks = param_picks_by_channel_types_or_names
    else:
        param_picks = None  

    # For continuous data 
    if param_epoched_data is False:
        
        # Load data
        data.load_data()

        # Bandpass, lowpass, or highpass filter
        data_filtered = data.filter(l_freq=param_l_freq, h_freq=param_h_freq, 
                                    picks=param_picks, filter_length=param_filter_length,
                                    l_trans_bandwidth=param_l_trans_bandwidth,
                                    h_trans_bandwidth=param_h_trans_bandwidth, n_jobs=param_n_jobs,
                                    method=param_method, iir_params=param_iir_params, phase=param_phase,
                                    fir_window=param_fir_window, fir_design=param_fir_design,
                                    skip_by_annotation=param_skip_by_annotation, pad=param_raw_pad)

    # For epoched data 
    else:

        # Bandpass, lowpass, or highpass filter
        data_filtered = data.filter(l_freq=param_l_freq, h_freq=param_h_freq, 
                                    picks=param_picks, filter_length=param_filter_length,
                                    l_trans_bandwidth=param_l_trans_bandwidth,
                                    h_trans_bandwidth=param_h_trans_bandwidth, n_jobs=param_n_jobs,
                                    method=param_method, iir_params=param_iir_params, phase=param_phase,
                                    fir_window=param_fir_window, fir_design=param_fir_design,
                                    skip_by_annotation=param_skip_by_annotation, pad=param_epoch_pad)

    # Save file
    data_filtered.save("out_dir_temporal_filtering/meg.fif", overwrite=True)

    return data_filtered


def _compute_snr(meg_file):
    # Compute the SNR

    # select only MEG channels and exclude the bad channels
    meg_file = meg_file.pick_types(meg=True, exclude='bads')

    # create fixed length events
    array_events = mne.make_fixed_length_events(meg_file, duration=10)

    # create epochs
    epochs = mne.Epochs(meg_file, array_events)

    # mean signal amplitude on each epoch
    epochs_data = epochs.get_data()
    mean_signal_amplitude_per_epoch = epochs_data.mean(axis=(1, 2))  # mean on channels and times

    # mean across all epochs and its std error
    mean_final = mean_signal_amplitude_per_epoch.mean()
    std_error_final = np.std(mean_signal_amplitude_per_epoch, ddof=1) / np.sqrt(
        np.size(mean_signal_amplitude_per_epoch))

    # compute SNR
    snr = mean_final / std_error_final

    return snr


def _generate_report(data_file_before, data_before_preprocessing, data_after_preprocessing, bad_channels,
                     comments_about_filtering, param_epoched_data, param_l_freq, param_h_freq, 
                     param_picks_by_channel_types_or_names, 
                     param_filter_length, param_picks_by_channel_indices,
                     param_l_trans_bandwidth, param_h_trans_bandwidth, param_n_jobs,
                     param_method, param_iir_params, param_phase, param_fir_window,
                     param_fir_design, param_skip_by_annotation, param_raw_pad, param_epoch_pad):
    # Generate a report

    # Instance of mne.Report
    report = mne.Report(title='Results of filtering ', verbose=True)

    # Put this info in html format #

    # Check if MaxFilter was already applied on the data #

    if data_before_preprocessing.info['proc_history']:
        sss_info = data_before_preprocessing.info['proc_history'][0]['max_info']['sss_info']
        tsss_info = data_before_preprocessing.info['proc_history'][0]['max_info']['max_st']
        if bool(sss_info) or bool(tsss_info) is True:
            message_channels = f'Bad channels have been interpolated during MaxFilter'
        else:
            message_channels = bad_channels
    else:
        message_channels = bad_channels

    # Give some info about the file before preprocessing
    sampling_frequency = data_before_preprocessing.info['sfreq']
    highpass = data_before_preprocessing.info['highpass']
    lowpass = data_before_preprocessing.info['lowpass']

    # Info on data #
    html_text_info = f"""<html>

        <head>
            <style type="text/css">
                table {{ border-collapse: collapse;}}
                td {{ text-align: center; border: 1px solid #000000; border-style: dashed; font-size: 15px; }}
            </style>
        </head>

        <body>
            <table width="50%" height="80%" border="2px">
                <tr>
                    <td>Input file: {data_file_before}</td>
                </tr>
                <tr>
                    <td>Epoched data: {param_epoched_data}</td>
                </tr>
                <tr>
                    <td>Bad channels: {message_channels}</td>
                </tr>
                <tr>
                    <td>Sampling frequency: {sampling_frequency}Hz</td>
                </tr>
                <tr>
                    <td>Highpass before preprocessing: {highpass}Hz</td>
                </tr>
                <tr>
                    <td>Lowpass before preprocessing: {lowpass}Hz</td>
                </tr>
            </table>
        </body>

        </html>"""

    # Add html to reports
    report.add_htmls_to_section(html_text_info, captions='Data recording features', section='Data info', replace=False)

    # Define param_picks
    if param_picks_by_channel_types_or_names is None and param_picks_by_channel_indices is not None:
        param_picks = param_picks_by_channel_indices
    elif param_picks_by_channel_types_or_names is not None and param_picks_by_channel_indices is None:
        param_picks = param_picks_by_channel_types_or_names
    else:
        param_picks = None    

    ## Plot figures for raw data ##
    if param_epoched_data is False:

        # Plot MEG signals in temporal domain
        fig_raw = data_before_preprocessing.pick(param_picks, exclude='bads').plot(duration=10, scalings='auto', butterfly=False,
                                                                                   show_scrollbars=False, proj=False, show=False)
        fig_raw_filtered = data_after_preprocessing.pick(param_picks, exclude='bads').plot(duration=10, scalings='auto',
                                                                                           butterfly=False,
                                                                                           show_scrollbars=False, 
                                                                                           proj=False, show=False)

        # Plot power spectral density
        fig_raw_psd = data_before_preprocessing.plot_psd(picks=param_picks, show=False)
        fig_raw_filtered_psd = data_after_preprocessing.plot_psd(picks=param_picks, show=False)

        # Add figures to report
        report.add_figs_to_section(fig_raw, captions='Signals before filtering', section='Temporal domain')
        report.add_figs_to_section(fig_raw_filtered, captions='Signals after filtering',
                                   comments=comments_about_filtering,
                                   section='Temporal domain')
        report.add_figs_to_section(fig_raw_psd, captions='Power spectral density before filtering',
                                   section='Frequency domain')
        report.add_figs_to_section(fig_raw_filtered_psd, captions='Power spectral density after filtering',
                                   comments=comments_about_filtering,
                                   section='Frequency domain')

        param_pad = param_raw_pad

    ## Plot figures for epoched data ##
    else:

        # Plot MEG signals in temporal domain
        fig_epoch = data_before_preprocessing.plot(picks=param_picks, scalings="auto", butterfly=False, show_scrollbars=False, show=False)
        fig_epoch_filtered = data_after_preprocessing.plot(picks=param_picks, scalings="auto", butterfly=False, show_scrollbars=False, show=False)

        # Plot power spectral density
        fig_epoch_psd = data_before_preprocessing.plot_psd(picks=param_picks, show=False)
        fig_epoch_filtered_psd = data_after_preprocessing.plot_psd(picks=param_picks, show=False)

        # Add figures to report
        report.add_figs_to_section(fig_epoch, captions='Signals before filtering', section='Temporal domain')
        report.add_figs_to_section(fig_epoch_filtered, captions='Signals after filtering',
                                   comments=comments_about_filtering,
                                   section='Temporal domain')
        report.add_figs_to_section(fig_epoch_psd, captions='Power spectral density before filtering',
                                   section='Frequency domain')
        report.add_figs_to_section(fig_epoch_filtered_psd, captions='Power spectral density after filtering',
                                   comments=comments_about_filtering,
                                   section='Frequency domain')

        param_pad = param_epoch_pad

    # # Info on SNR
    # html_text_snr = f"""<html>

    # <head>
    #     <style type="text/css">
    #         table {{ border-collapse: collapse;}}
    #         td {{ text-align: center; border: 1px solid #000000; border-style: dashed; font-size: 15px; }}
    #     </style>
    # </head>

    # <body>
    #     <table width="50%" height="80%" border="2px">
    #         <tr>
    #             <td>SNR before filtering: {snr_before}</td>
    #         </tr>
    #         <tr>
    #             <td>SNR after filtering: {snr_after}</td>
    #         </tr>
    #     </table>
    # </body>

    # </html>"""

    ## Values of the parameters of the App ## 
    mne_version = mne.__version__

    # Put this info in html format # 
    html_text_parameters = f"""<html>

    <head>
        <style type="text/css">
            table {{ border-collapse: collapse;}}
            td {{ text-align: center; border: 1px solid #000000; border-style: dashed; font-size: 15px; }}
        </style>
    </head>

    <body>
        <table width="50%" height="80%" border="2px">
            <tr>
                <td>Temporal filtering: {comments_about_filtering}</td>
            </tr>
            <tr>
                <td>Types or names of channels to include: {param_picks_by_channel_types_or_names}</td>
            </tr>
            <tr>
                <td>Indices of channels to include: {param_picks_by_channel_indices}</td>
            </tr>
            <tr>
                <td>Filter length: {param_filter_length}</td>
            </tr>
            <tr>
                <td>Width of the transition band at the low cut-off frequency: {param_l_trans_bandwidth}</td>
            </tr>
            <tr>
                <td>Width of the transition band at the high cut-off frequency: {param_l_trans_bandwidth}</td>
            </tr>
            <tr>
                <td>Number of jobs to run in parallel: {param_n_jobs}</td>
            </tr>
            <tr>
                <td>Method: {param_method}</td>
            </tr>
            <tr>
                <td>IIR parameters: {param_iir_params}</td>
            </tr>
            <tr>
                <td>Phase: {param_phase}</td>
            </tr>
            <tr>
                <td>FIR window: {param_fir_window}</td>
            </tr>
            <tr>
                <td>FIR design: {param_fir_design}</td>
            </tr>
            <tr>
                <td>Skip by annotation: {param_skip_by_annotation}</td>
            </tr>
            <tr>
                <td>Type of padding: {param_pad}</td>
            </tr>
            <tr>
                <td>MNE version used: {mne_version}</td>
            </tr>
        </table>
    </body>

    </html>"""

    # Add html to report
    report.add_htmls_to_section(html_text_parameters , captions='Summary filtering applied',
                                section='Filtering info', replace=False)
    # report.add_htmls_to_section(html_text_snr, captions='Signal to noise ratio', section='Signal to noise ratio',
    #                             replace=False)

    # Save report
    report.save('out_dir_report/report_filtering.html', overwrite=True)


def main():

    # Generate a json.product to display messages on Brainlife UI
    dict_json_product = {'brainlife': []}

    # Load inputs from config.json
    with open('config.json') as config_json:
        config = json.load(config_json)

    # Read the files
    data_file = config.pop('mne')
    if config['param_epoched_data'] is False:
        data = mne.io.read_raw_fif(data_file, allow_maxshield=True)
    else:
        data = mne.read_epochs(data_file)

    # Read and save optional files
    config, cross_talk_file, calibration_file, events_file, head_pos_file, channels_file, destination = helper.read_optional_files(config, 'out_dir_temporal_filtering')
    
    # Convert empty strings values to None
    config = helper.convert_parameters_to_None(config)


    ## Convert parameters ## 

    # Deal with param_picks_by_channel_indices parameter #
    # Convert param_picks_by_channel_indices into slice when the App is run locally and on BL
    picks = config['param_picks_by_channel_indices']
    if isinstance(picks, str) and picks.find(",") != -1 and picks.find("[") == -1 and picks is not None:
        picks = list(map(int, picks.split(', ')))
        if len(picks) == 2:
            config['param_picks_by_channel_indices'] = slice(picks[0], picks[1])
        elif len(picks) == 3:
            config['param_picks_by_channel_indices'] = slice(picks[0], picks[1], picks[2])
        else:
            value_error_message = f"If you want to select channels using a slice, you must give two or three elements."
            raise ValueError(value_error_message)

    # Convert param_picks_by_channel_indices into a list of integers when the App is run on BL
    if isinstance(picks, str) and picks.find(",") != -1 and picks.find("[") != -1 and picks is not None:
        picks = picks.replace('[', '')
        picks = picks.replace(']', '')
        config['param_picks_by_channel_indices'] = list(map(int, picks.split(', ')))

    # Deal with param_picks_by_channel_types_or_name parameter #
    # Convert param_picks_by_channel_types_or_names into a list of string when the App is run on BL
    picks = config['param_picks_by_channel_types_or_names']
    if isinstance(picks, str) and picks.find("[") != -1 and picks is not None:
        picks = picks.replace('[', '')
        picks = picks.replace(']', '')
        config['param_picks_by_channel_types_or_names'] = list(map(str, picks.split(', ')))

    # Deal with filter_length parameter on BL # 
    # Convert param_filter_length into int if not auto and not a length in time when the App is run on BL
    if config['param_filter_length'] != "auto" and config['param_filter_length'].find("s") == -1:
        config['param_filter_length'] = int(config['param_filter_length'])

    # Deal with param_l_trans_bandwidth parameter on BL # 
    # Convert param_l_trans_bandwidth into a float if not auto when the App is run on BL
    if isinstance(config['param_l_trans_bandwidth'], str) and config['param_l_trans_bandwidth'] != "auto":
         config['param_l_trans_bandwidth'] = float(config['param_l_trans_bandwidth'])

    # Deal with param_h_trans_bandwidth parameter on BL #
    # Convert param_h_trans_bandwidth into a float if not auto when the App is run on BL
    if isinstance(config['param_h_trans_bandwidth'], str) and config['param_h_trans_bandwidth'] != "auto":
         config['param_h_trans_bandwidth'] = float(config['param_h_trans_bandwidth'])

    # Deal with param_n_jobs parameter on BL # 
    # Convert param n jobs into an int if not cuda
    if config['param_n_jobs'] != 'cuda':
        config['param_n_jobs']  = int(config['param_n_jobs'])

    # Deal with skip_by_annotation parameter #
    # Convert param_mag_scale into a list of strings when the app runs on BL
    skip_by_an = config['param_skip_by_annotation']
    if skip_by_an == "[]":
        skip_by_an = []
    elif isinstance(skip_by_an, str) and skip_by_an.find("[") != -1 and skip_by_an != "[]": 
        skip_by_an = skip_by_an.replace('[', '')
        skip_by_an = skip_by_an.replace(']', '')
        skip_by_an = list(map(str, skip_by_an.split(', ')))         
    config['param_skip_by_annotation'] = skip_by_an 

    
    ## Info message about filtering ## 

    # Band pass filter
    if config['param_l_freq'] is not None and config['param_h_freq'] is not None:
        comments_about_filtering = f'Data was filtered between ' \
                                   f'{config["param_l_freq"]} ' \
                                   f'and {config["param_h_freq"]}Hz.'
        dict_json_product['brainlife'].append({'type': 'info', 'msg': comments_about_filtering})
        filter_type = "band-pass"

    # Lowpass filter
    elif config['param_l_freq'] is None and config['param_h_freq'] is not None:
        comments_about_filtering = f'Lowpass filter was applied at {config["param_h_freq"]}Hz.'
        dict_json_product['brainlife'].append({'type': 'info', 'msg': comments_about_filtering})
        filter_type = "low-pass"

    # Highpass filter
    elif config['param_l_freq'] is not None and config['param_h_freq'] is None:
        comments_about_filtering = f'Highpass filter was applied at {config["param_l_freq"]}Hz.'
        dict_json_product['brainlife'].append({'type': 'info', 'msg': comments_about_filtering})
        filter_type = "high-pass"

    # Raise an exception if both param_filter_l_freq and param_filter_h_freq are None
    elif config['param_l_freq'] is None and config["param_h_freq"] is None:
        value_error_message = f'You must specify a value for param_l_freq or param_h_freq, ' \
                              f"they can't both be set to None."
        # Raise exception
        raise ValueError(value_error_message)

    # Channels.tsv must be BIDS compliant
    if channels_file is not None:
        user_warning_message_channels = f'The channels file provided must be ' \
                                        f'BIDS compliant and the column "status" must be present. ' 
        warnings.warn(user_warning_message_channels)
        dict_json_product['brainlife'].append({'type': 'warning', 'msg': user_warning_message_channels})
        # Udpate data.info['bads'] with info contained in channels.tsv
        data, user_warning_message_channels = helper.update_data_info_bads(data, channels_file)
        if user_warning_message_channels is not None: 
            warnings.warn(user_warning_message_channels)
            dict_json_product['brainlife'].append({'type': 'warning', 'msg': user_warning_message_channels})

    
    # Keep bad channels in memory
    bad_channels = data.info['bads']

    # Delete keys values in config.json when this app is executed on Brainlife
    kwargs = helper.define_kwargs(config) 

    # Apply temporal filtering
    data_copy = data.copy()
    data_filtered = temporal_filtering(data_copy, **kwargs)
    del data_copy

    # Success message in product.json    
    dict_json_product['brainlife'].append({'type': 'success', 'msg': 'Filtering was applied successfully.'})

    # Compute SNR
    #snr_before = _compute_snr(data)
    #snr_after = _compute_snr(data_filtered)

    # Generate a report
    _generate_report(data_file, data, data_filtered, bad_channels, comments_about_filtering, **kwargs)

    # Save the dict_json_product in a json file
    with open('product.json', 'w') as outfile:
        json.dump(dict_json_product, outfile)


if __name__ == '__main__':
    main()
