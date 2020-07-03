# inertial-sensor-processing

Processing of inertial sensor data for intake gesture detection.

## Features

Implemented processing steps include mirroring, gravity removal, smoothing, and standardization.
Export as `csv` or `tfrecord` files.
Specific labels can be selected by including a specification file.

Supported datasets currently include:
- [OREBA-DIS and OREBA-SHA](http://www.newcastle.edu.au/oreba)
- [Clemson Cafeteria](http://cecas.clemson.edu/~ahoover/cafeteria/)
- [Food Intake Cycle (FIC)](https://mug.ee.auth.gr/intake-cycle-detection/)

## Usage

Make sure that all requirements are fulfilled

```
$ pip install -r requirements.txt
```

Then call `main.py`.

```
$ python main.py --src_dir=OREBA-DIS
```

The following flags can be used to specify the settings:

| Argument | Description | Default |
| --- | --- | --- |
| --src_dir | Directory to search for data | OREBA-DIS |
| --exp_dir | Directory for data export | Export |
| --dataset | Which dataset is used {OREBA-DIS, OREBA-SHA, Clemson, or FIC} | OREBA-DIS |
| --sampling_rate | Sampling rate of exported signals in Hz | 64 |
| --use_vis | If True, enable visualization | False |
| --use_gravity_removal | If True, remove gravity during preprocessing | True |
| --use_smoothing | If True, apply smoothing during preprocessing | False |
| --use_standardization | If True, apply standardization during preprocessing | True |
| --smoothing_window_size | Size of the smoothing window [number of frames] | 1 |
| --smoothing_order | The polynomial used in Savgol filter | 1 |
| --smoothing_mode | Smoothing mode {medfilt, savgol_filter, moving_average} | moving_average |
| --exp_mode | Write file for publication or development {pub, dev} | dev |
| --exp_uniform | Convert all dominant hands to right and all non-dominant hands to left | True |
| --exp_format | Format for export {csv, tfrecord} | csv |
| --label_spec | Filename of label specification within src_dir | labels.xml |
| --label_spec_inherit | Inherit label specification for sublabels (if label not included, always keep sublabels as Idle) | True |
| --dom_hand_spec | Filename containing the dominant hand info | dominant_hand.csv |
| --organise_data | If True, organise data in train, valid, test subfolders | False |
| --organise_dir | Directory to copy train, val and test sets using data organiser | Organised |
| --organise_subfolders | Create sub folder per each file in validation and test set | False |

## Label specfication

Control what labels are included by selecting or editing the appropriate `label_spec` file.
This only applies to the OREBA and Clemson datasets.
Templates are available in the `label_spec` directory.
