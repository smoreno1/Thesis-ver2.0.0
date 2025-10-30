import numpy as np

'''
mesoSPIM configuration file.

Use this file as a starting point to set up all mesoSPIM hardware by replacing the 'Demo' designations
with real hardware one-by-one. Make sure to rename your new configuration file to a different filename
(The extension has to be .py).
'''

'''
Dark mode: Renders the UI dark
'''
ui_options = {'dark_mode' : True, # Dark mode: Renders the UI dark if enabled
              'enable_x_buttons' : True, # Here, specific sets of UI buttons can be disabled
              'enable_y_buttons' : True,
              'enable_z_buttons' : True,
              'enable_f_buttons' : True,
              'enable_f_zero_button' : True, # set to False if objective change requires F-stage movement (e.g. mesoSPIM v6-Revolver), for safety reasons
              'enable_rotation_buttons' : True,
              'enable_loading_buttons' : True,
              'flip_XYZFT_button_polarity': (True, False, False, False, False), # flip the polarity of the stage buttons (X, Y, Z, F, Theta)
              'button_sleep_ms_xyzft' : (250, 0, 250, 0, 0), # step-motion buttons disabled for N ms after click. Prevents stage overshooting outside of safe limits, for slow stages.
              'window_pos': (100, 100), # position of the main window on the screen, top left corner.
              'usb_webcam_ID': None, # open USB web-camera (if available): None,  0 (first cam), 1 (second cam), ...
              'flip_auto_LR_illumination': False, # flip the polarity of the "Auto L/R illumination" button in Acquisition Manager
               }

logging_level = 'DEBUG' # 'DEBUG' for ultra-detailed, 'INFO' for general logging level

'''
Waveform output for Galvos, ETLs etc.
'''
waveformgeneration = 'NI' # 'DemoWaveFormGeneration' or 'NI'

'''
Card designations need to be the same as in NI MAX, if necessary, use NI MAX
to rename your cards correctly.

The new mesoSPIM configuration (benchtop-inspired) uses one card (PXI6733) and allows up to 4 laser channels.

Physical channels must be connected in certain order:
- 'galvo_etl_task_line' takes Galvo-L, Galvo-R, ETL-L, ETL-R
(e.g. value 'PXI6259/ao0:3' means Galvo-L on ao0, Galvo-R on ao1, ETL-L on ao2, ETL-R on ao3)

- 'laser_task_line' takes laser modulation, lasers sorted in increasing wavelength order,
(e.g. value 'PXI6733/ao4:7' means '405 nm' connected to ao4, '488 nm' to ao5, etc.)
'''

acquisition_hardware = {'master_trigger_out_line' : 'PXI6259/port0/line0',
                        'camera_trigger_source' : '/PXI6259/PFI0',
                        'camera_trigger_out_line' : '/PXI6259/ctr0',
                        'galvo_etl_task_line' : 'PXI6259/ao0:3',
                        'galvo_etl_task_trigger_source' : '/PXI6259/PFI0',
                        'laser_task_line' :  'PXI6733/ao0:3',
                        'laser_task_trigger_source' : '/PXI6259/PFI0'}

sidepanel = 'Demo' #'Demo' or 'FarmSimulator', deprecated

'''
Digital laser enable lines
'''

laser = 'NI' # 'Demo' or 'NI'

''' The `laserdict` specifies laser labels of the GUI and their digital modulation channels.
Keys are the laser designation that will be shown in the user interface
Values are DO ports used for laser ENABLE digital signal.
Critical: entries must be sorted in the increasing wavelength order: 405, 488, etc.
'''
laserdict = {'405 nm': 'PXI6733/port0/line4',
             '488 nm': 'PXI6733/port0/line2',
             '532 nm': 'PXI6733/port0/line5',
             '647 nm': 'PXI6733/port0/line3',
             }

''' Laser blanking indicates whether the laser enable lines should be set to LOW between
individual images or stacks. This is helpful to avoid laser bleedthrough between images caused by insufficient
modulation depth of the analog input (even at 0V, some laser light is still emitted).
'''
laser_blanking = 'images' # if 'images', laser is off before and after every image; if 'stacks', before and after each stack.

'''
Shutter configuration
If shutterswitch = True:
    'shutter_left' is the general shutter
    'shutter_right' is the left/right switch (Right==True)

If shutterswitch = False or missing:
    'shutter_left' and 'shutter_right' are two independent shutters.
'''
shutter = 'NI' # 'Demo' or 'NI'
shutterswitch = False # see legend above
shutteroptions = ('Left', 'Right') # Shutter options of the GUI
shutterdict = {'shutter_left' : 'PXI6259/port0/line8', # left (general) shutter
              'shutter_right' : 'PXI6259/port0/line5'} # flip mirror or right shutter, depending on physical configuration

'''
Camera configuration

camera = 'DemoCamera' # 'DemoCamera' or 'HamamatsuOrca' or 'Photometrics'


'''

camera = 'Photometrics'

camera_parameters = {'x_pixels' : 3200,
                     'y_pixels' : 3200,
                     'x_pixel_size_in_microns' : 6.5,
                     'y_pixel_size_in_microns' : 6.5,
                     'subsampling' : [1,2,4],
                     'speed_table_index': 0,
                     'exp_mode' : 'Edge Trigger', # Lots of options in PyVCAM --> see constants.py
                     'readout_port': 2,
                     'gain_index': 1,
                     'exp_out_mode': 4, # 4: line out
                     'binning' : '1x1',
                     'scan_mode' : 1, # Scan mode options: {'Auto': 0, 'Line Delay': 1, 'Scan Width': 2}
                     'scan_direction' : 0, # Scan direction options: {'Down': 0, 'Up': 1, 'Down/Up Alternate': 2}
                     'scan_line_delay' : 13, # Works with 225 ms sweeptime
                    }


binning_dict = {'1x1': (1,1), '2x2':(2,2), '4x4':(4,4)}

'''
Stage configuration
'''

'''
The stage_parameter dictionary defines the general stage configuration, initial positions,
and safety limits. The rotation position defines a XYZ position (in absolute coordinates)
where sample rotation is safe. Additional hardware dictionaries (e.g. pi_parameters)
define the stage configuration details.
All positions are absolute.

'stage_type' option:
ASI stages, 'stage_type' : 'TigerASI', 'MS2000ASI'
PI stages, 'stage_type' : 'PI' or 'PI_1controllerNstages' (equivalent), 'PI_NcontrollersNstages'
Mixed stages, 'stage_type' : 'PI_rot_and_Galil_xyzf', 'GalilStage', 'PI_f_rot_and_Galil_xyz', 'PI_rotz_and_Galil_xyf', 'PI_rotzf_and_Galil_xy',
'''

stage_parameters = {'stage_type' : 'PI_1controllerNstages', # one of 'DemoStage', , 'PI_NcontrollersNstages', 'TigerASI', etc, see above
                    'startfocus': 20000,
                    'y_load_position': 90000,
                    'y_unload_position': 2000,
                    'x_max' : 40000,
                    'x_min' : 8000,
                    'y_max' : 150000,
                    'y_min' : 2,
                    'z_max' : 45000,
                    'z_min' : 23000,
                    'f_max' : 68000,
                    'f_min' : 40000,
                    'theta_max' : 360,
                    'theta_min' : -360,
                    'x_rot_position':27000,
                    'y_rot_position': 99000,
                    'z_rot_position': 34000,
                    }

pi_parameters = {'controllername' : 'C-884',
                 'stages' : ('L-509.20DG10','L-509.40DG10','L-509.20DG10','M-060.DG','M-406.4PD','NOSTAGE'),
                 'refmode' : ('FRF',),
                 'serialnum' : ('123034131'),
                 }


'''
Filterwheel configuration
For a DemoFilterWheel, no COMport needs to be specified.
For a Ludl Filterwheel, a valid COMport is necessary. Ludl marking 10 = position 0.
For a Dynamixel FilterWheel, valid baudrate and servoi_id are necessary.
'''
filterwheel_parameters = {'filterwheel_type' : 'ZWO', # 'Demo', 'Ludl', 'Sutter', 'Dynamixel', 'ZWO'
                          'COMport' : 'COM3', # irrelevant for 'ZWO'
                          'baudrate' : 115200, # relevant only for 'Dynamixel'
                          'servo_id' :  1, # relevant only for 'Dynamixel'
                          }
'''
filterdict contains filter labels and their positions. The valid positions are:
For Ludl: 0, 1, 2, 3, .., 9, i.e. position ids (int)
For Dynamixel: servo encoder counts, e.g. 0 for 0 deg, 1024 for 45 deg (360 deg = 4096 counts, or 11.377 counts/deg).
Dynamixel encoder range in multi-turn mode: -28672 .. +28672 counts.
For ZWO EFW Mini 5-slot wheel: positions 0, 1, .. 4.
'''
filterdict = {'445nm_45nm' : 0,
              '514nm_30nm' : 1, # Every config should contain at least this
              '575nm_59nm' : 2,
              '655nm_LP' : 3,
              '620nm_14nm': 4,
              'Empty' : 5,} # Dictionary labels must be unique!

'''
Zoom configuration
For the 'Demo', 'servo_id', 'COMport' and 'baudrate' do not matter.
For a 'Dynamixel' servo-driven zoom, 'servo_id', 'COMport' and 'baudrate' (default 1000000) must be specified
For 'Mitu' (Mitutoyo revolver), 'COMport' and 'baudrate' (default 9600) must be specified
'''
zoom_parameters = {'zoom_type' : 'Mitu', # 'Demo', 'Dynamixel', or 'Mitu'
                   'COMport' : 'COM1',
                   'baudrate' : 9600,
                   }

'''
The keys in the zoomdict define what zoom positions are displayed in the selection box
(combobox) in the user interface.
'''

'''
The 'Mitu' (Mitutoyo revolver) positions
'''
zoomdict = {'2x': 'A',
            '5x': 'B',
            '10x': 'C',
            }

'''
Pixelsize in micron
'''
pixelsize = {'2x' : 3.275,
            '5x' : 1.31,
            '10x' : 0.655,}

'''
 HDF5 parameters, if this format is used for data saving (optional).
Downsampling and compression slows down writing by 5x - 10x, use with caution.
Imaris can open these files if no subsampling and no compression is used.
'''
hdf5 = {'subsamp': ((1, 1, 1),), #((1, 1, 1),) no subsamp, ((1, 1, 1), (1, 4, 4)) for 2-level (z,y,x) subsamp.
        'compression': None, # None, 'gzip', 'lzf'
        'flip_xyz': (True, True, False), # match BigStitcher coordinates to mesoSPIM axes.
        'transpose_xy': False, # in case X and Y axes need to be swapped for the correct tile positions
        }

buffering = {'use_ram_buffer': True, # If True, the data is buffered in RAM before writing to disk. If False, data is written to disk immediately after each frame
             'percent_ram_free': 20, # If use_ram_buffer is True and once the free RAM is below this value, the data is written to disk.
             }
'''
Rescale the galvo amplitude when zoom is changed
For example, if 'galvo_l_amplitude' = 1 V at zoom '1x', it will ve 2 V at zoom '0.5x'
'''
scale_galvo_amp_with_zoom = True

'''
Initial acquisition parameters
Used as initial values after startup
When setting up a new mesoSPIM, make sure that:
* 'max_laser_voltage' is correct (5 V for Toptica MLEs, 10 V for Omicron SOLE)
* 'galvo_l_amplitude' and 'galvo_r_amplitude' (in V) are correct (not above the max input allowed by your galvos)
* all the filepaths exist
* the initial filter exists in the filter dictionary above
'''
startup = {
'state' : 'init', # 'init', 'idle' , 'live', 'snap', 'running_script'
'samplerate' : 100000,
'sweeptime' : 0.225,
'position' : {'x_pos':0,'y_pos':1000,'z_pos':2000,'f_pos':5000,'theta_pos':180},
'ETL_cfg_file' : 'config/etl_parameters/ETL-parameters.csv',
'folder' : 'F:/tmp/',
'snap_folder' : 'F:/tmp/',
'file_prefix' : '',
'file_suffix' : '000001',
'zoom' : '5x',
'pixelsize' : 1.0,
'laser' : '488 nm',
'max_laser_voltage':5,
'intensity' : 10,
'shutterstate':False, # Is the shutter open or not?
'shutterconfig':'Right', # Can be "Left", "Right","Both","Interleaved"
'laser_interleaving':False,
'filter' : 'Empty',
'etl_l_delay_%' : 7.5,
'etl_l_ramp_rising_%' : 85,
'etl_l_ramp_falling_%' : 2.5,
'etl_l_amplitude' : 0.7,
'etl_l_offset' : 2.3,
'etl_r_delay_%' : 2.5,
'etl_r_ramp_rising_%' : 5,
'etl_r_ramp_falling_%' : 85,
'etl_r_amplitude' : 0.65,
'etl_r_offset' : 2.36,
'galvo_l_frequency' : 99.9,
'galvo_l_amplitude' : 2.5,
'galvo_l_offset' : 0,
'galvo_l_duty_cycle' : 50,
'galvo_l_phase' : np.pi/2,
'galvo_r_frequency' : 99.9,
'galvo_r_amplitude' : 1.2,
'galvo_r_offset' : 0.75,
'galvo_r_duty_cycle' : 50,
'galvo_r_phase' : np.pi/2,
'laser_l_delay_%' : 10,
'laser_l_pulse_%' : 87,
'laser_l_max_amplitude_%' : 100,
'laser_r_delay_%' : 10,
'laser_r_pulse_%' : 87,
'laser_r_max_amplitude_%' : 100,
'camera_delay_%' : 10,
'camera_pulse_%' : 1,
'camera_exposure_time':0.02,
'camera_line_interval':0.000075, # Hamamatsu-specific parameter
'camera_display_live_subsampling': 1,
#'camera_display_snap_subsampling': 1, #deprecated
'camera_display_acquisition_subsampling': 1,
'camera_binning':'1x1',
'camera_sensor_mode':'ASLM', # Hamamatsu-specific parameter
'average_frame_rate': 2.5,
}
