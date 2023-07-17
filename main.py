# this app allows filtering raw data.

import os
import mne
import json
import helper
import matplotlib.pyplot as plt

#workaround for -- _tkinter.TclError: invalid command name ".!canvas"
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load brainlife config.json
with open('config.json','r') as config_f:
    config = helper.convert_parameters_to_None(json.load(config_f))

# == LOAD DATA ==
fname = config['mne']
raw = mne.io.read_raw_fif(fname, preload=True)

if config['l_freq'] is not None:
    raw = raw.filter(l_freq=config['l_freq'], h_freq=None)

ica= ICA(n_components=config['n_components'], noise_cov=config['noise_cov'],
                      random_state=config['random_state'], method=config['method'],
                      fit_params=config['fit_params'], max_iter=config['max_iter'],
                      allow_ref_meg=config['allow_ref_meg'])

ica.fit(raw)
ica

explained_var_ratio = ica.get_explained_variance_ratio(raw)
for channel_type, ratio in explained_var_ratio.items():
    print(
        f'Fraction of {channel_type} variance explained by all components: '
        f'{ratio}'
    )


ica.save('out_dir/ica.fif',overwrite=True)

plt.figure(1)
ica.plot_components()
plt.savefig(os.path.join('out_figs','components_topo.png'))
plt.close()

# now for each figure poped by ica.plot_properties, save it
fs = ica.plot_properties(raw, picks=config['picks_to_plot'])
[f.savefig(os.path.join('out_figs','component_'+str(i)+'.png')) for i, f in enumerate(fs)]

# ica.plot_sources(raw)
# plt.savefig(os.path.join('out_figs','components_timecourse.png'))

eog_evoked = create_eog_epochs(raw, ch_name = config['eog_ch']).average()
ecg_evoked = create_ecg_epochs(raw, ch_name = config['ecg_ch']).average()
eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name = config['eog_ch'])
ecg_indices, ecg_scores = ica.find_bads_ecg(raw, ch_name = config['ecg_ch'])

# combine eog_indices and ecg_indices into one list
ica.exclude = list(set(eog_indices + ecg_indices))

report = mne.Report(title='ICA')
report.add_ica(ica, 'Fitted ICA', inst = raw, eog_evoked = eog_evoked, ecg_evoked = ecg_evoked,
               ecg_scores = ecg_scores, eog_scores = eog_scores)

report.save('out_report/report_ica.html', overwrite=True)



