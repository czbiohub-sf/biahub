#%% import libraries

import numpy as np
from pycromanager import Bridge, Acquisition, multi_d_acquisition_events

#%% initialize bridge

bridge = Bridge()
mmc = bridge.get_core()
mmStudio = bridge.get_studio()

#%% import positions from MM

mm_pos_list = mmStudio.get_position_list_manager().get_position_list()
number_of_positions = mm_pos_list.get_number_of_positions()

xy_position_list = [[mm_pos_list.get_position(i).get_x(),
                     mm_pos_list.get_position(i).get_y()]
                    for i in range(number_of_positions)]

z_position_list = [mm_pos_list.get_position(i).get('ZDrive').get1_d_position()
                   for i in range(number_of_positions)]

#%% define imaging settings

data_directory = r'E:\2022_07_08 zebrafish imaging'
data_name = 'fish2_withGFP'

# fluorescence channels
epi_channels = [{'group': 'Master Channel', 'config': 'Epi-GFP'},
                {'group': 'Master Channel', 'config': 'Epi-DSRED'}]
epi_exposure = [150, 80]  # in ms

# LF channels, 5-state algorithm
lf_channels = [{'group': 'Master Channel', 'config': f'State{i}'} for i in range(5)]
lf_exposure = 5

# relative z, acquired using piezo stage
z_start = -10
z_end = 20
z_step = 0.3
z_range = np.arange(z_start, z_end + z_step, z_step)

num_time_points = 15
time_interval_s = 0  # in seconds

#%% prepare for acquisition

# setup Prime BSI Express camera
mmc.set_property('Prime BSI Express', 'Binning', '2x2')
mmc.set_property('Prime BSI Express', 'ReadoutRate', '100MHz 16bit')
mmc.set_property('Prime BSI Express', 'Gain', '2-CMS')

# setup Oryx camera
mmc.set_property('Oryx', 'Frame Rate Control Enabled', '1')
mmc.set_property('Oryx', 'Frame Rate', '10')
mmc.set_property('Oryx', 'Pixel Format', 'Mono12p')
# mmc.set_roi('Oryx', 100, 0, 1024, 1024)

# setup TriggerScope
mmc.set_property('TS_DAC01', 'Sequence', 'On')
mmc.set_property('TS_DAC02', 'Sequence', 'On')
mmc.set_property('TS_DAC05', 'Sequence', 'On')

# set Focus device to piezo drive
mmc.set_property('Core', 'Focus', 'PiezoStage:Q:35')

# setup autofocus
autofocus_manager = mmStudio.get_autofocus_manager()
autofocus_manager.set_autofocus_method_by_name('PFS')

#%% generate acquisition events

events = []
for t_idx in range(num_time_points):
    for p_idx, p in enumerate(xy_position_list):

        # epi channels, z-first
        for channel, exp in zip(epi_channels, epi_exposure):
            for z_idx, z in enumerate(z_range):
                events.append({
                    'axes': {'time': t_idx, 'position': p_idx, 'z': z_idx},
                    'min_start_time': t_idx * time_interval_s,
                    'x': p[0],
                    'y': p[1],
                    'z': z,
                    'channel': channel,
                    'exposure': exp,
                })

        # lf positions, channel first - channel sequencing does not work
        for z_idx, z in enumerate(z_range):
            for channel in lf_channels:
                events.append({
                    'axes': {'time': t_idx, 'position': p_idx, 'z': z_idx},
                    'min_start_time': t_idx * time_interval_s,
                    'x': p[0],
                    'y': p[1],
                    'z': z,
                    'channel': channel,
                    'exposure': lf_exposure,
                })


#%% hook functions

def post_hardware_hook(event, bridge, event_queue):

    if 'axes' in event.keys():  # some events only modify hardware
        if not hasattr(post_hardware_hook, 'pos_index'):
            post_hardware_hook.pos_index = -1

        # Only run hook function at new positions (including the first one)
        pos_index = event['axes']['position']
        if pos_index != post_hardware_hook.pos_index:
            # get the Micro-Manager core
            mmc = bridge.get_core()

            # apply ZDrive position
            mmc.set_position('ZDrive', z_position_list[pos_index])

        # Keep track of last time point on which autofocus ran
        post_hardware_hook.pos_index = pos_index

    return event

#%%  acquire data

# turn on autoshutter
mmc.set_auto_shutter(True)

# turn off live preview
snap_live_manager = mmStudio.get_snap_live_manager()
if snap_live_manager.is_live_mode_on():
    snap_live_manager.set_live_mode_on(False)

# acquire data
# with Acquisition(directory=data_directory, name=data_name) as acq:
#     acq.acquire(events, keep_shutter_open=True)

with Acquisition(directory=data_directory, name=data_name, post_camera_hook_fn=post_hardware_hook) as acq:
    acq.acquire(events, keep_shutter_open=True)
