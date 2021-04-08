from __future__ import print_function, absolute_import
import numpy as np
import cv2
from pyaer import libcaer
from pyaer.davis import DAVIS
from pyaer.dvxplorer import DVXPLORER

import torch
from utils.loading_utils import load_model, get_device
import numpy as np
import argparse
import pandas as pd
from utils.event_readers import FixedSizeEventReader, FixedDurationEventReader
from utils.inference_utils import events_to_voxel_grid, events_to_voxel_grid_pytorch
from utils.timers import Timer
import time
from image_reconstructor import ImageReconstructor
from options.inference_options import set_inference_options



if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Evaluating a trained network')
    parser.add_argument('-c', '--path_to_model', required=True, type=str,
                        help='path to model weights')
    # parser.add_argument('-i', '--input_file', required=True, type=str)
    parser.add_argument('--fixed_duration', dest='fixed_duration', action='store_true')
    parser.set_defaults(fixed_duration=False)
    parser.add_argument('-N', '--window_size', default=None, type=int,
                        help="Size of each event window, in number of events. Ignored if --fixed_duration=True")
    # parser.add_argument('-T', '--window_duration', default=33.33, type=float,
                        # help="Duration of each event window, in milliseconds. Ignored if --fixed_duration=False")
    parser.add_argument('--num_events_per_pixel', default=0.35, type=float,
                        help='in case N (window size) is not specified, it will be \
                              automatically computed as N = width * height * num_events_per_pixel')
    parser.add_argument('--skipevents', default=0, type=int)
    parser.add_argument('--suboffset', default=0, type=int)
    parser.add_argument('--compute_voxel_grid_on_cpu', dest='compute_voxel_grid_on_cpu', action='store_true')
    parser.set_defaults(compute_voxel_grid_on_cpu=False)

    set_inference_options(parser)

    args = parser.parse_args()

    # Read sensor size from the first first line of the event file
    # path_to_events = args.input_file

    # header = pd.read_csv(path_to_events, delim_whitespace=True, header=None, names=['width', 'height'],
    #                      dtype={'width': np.int, 'height': np.int},
    #                      nrows=1)
    # width, height = header.values[0]
    width,height = 320,240
    print('Sensor size: {} x {}'.format(width, height))

    # Load model
    model = load_model(args.path_to_model)
    device = get_device(args.use_gpu)

    model = model.to(device)
    model.eval()

    reconstructor = ImageReconstructor(model, height, width, model.num_bins, args)

    """ Read chunks of events using Pandas """

    # Loop through the events and reconstruct images
    N = args.window_size
    if not args.fixed_duration:
        N = int(width * height * args.num_events_per_pixel)
        print('Will use {} events per tensor (automatically estimated with num_events_per_pixel={:0.2f}).'.format(N, args.num_events_per_pixel))
        # if N is None:
        #     N = int(width * height * args.num_events_per_pixel)
        #     print('Will use {} events per tensor (automatically estimated with num_events_per_pixel={:0.2f}).'.format(
        #         N, args.num_events_per_pixel))
        # else:
        #     print('Will use {} events per tensor (user-specified)'.format(N))
        #     mean_num_events_per_pixel = float(N) / float(width * height)
        #     if mean_num_events_per_pixel < 0.1:
        #         print('!!Warning!! the number of events used ({}) seems to be low compared to the sensor size. \
        #             The reconstruction results might be suboptimal.'.format(N))
        #     elif mean_num_events_per_pixel > 1.5:
        #         print('!!Warning!! the number of events used ({}) seems to be high compared to the sensor size. \
        #             The reconstruction results might be suboptimal.'.format(N))

    initial_offset = args.skipevents
    sub_offset = args.suboffset
    start_index = initial_offset + sub_offset
    print(initial_offset,sub_offset,start_index)



    # device1 = DAVIS(noise_filter=False)
    device1 = DVXPLORER()
    device1.start_data_stream()
    # load new config
    # device1.set_bias_from_json("./scripts/configs/davis240c_config.json")
    device1.set_bias_from_json("./scripts/configs/dvxplorer_config.json")
    
    while True:
        try:
            (pol_events, num_pol_event,
            special_events, num_special_event,
            imu_events, num_imu_event) = \
                device1.get_event("events")
            # (pol_events, num_pol_event,
            #  special_events, num_special_event,
            #  frames_ts, frames, imu_events,
            #  num_imu_event) = device1.get_event("events")
            if num_pol_event != 0:
                # events = pol_events[:,1:]
                # time = pol_events[:,0]

                pol_events = pol_events.astype(np.float32)
                event_window = pol_events
                event_window[:,0] = event_window[:,0]/1e6
                # event_window = event_window[0:5000]
                # print(event_window.shape)
                # print(event_window.shape)


                with Timer('Processing entire dataset'):
                    last_timestamp = event_window[-1, 0]
                    # print(last_timestamp)


                    with Timer('Building event tensor'):
                        if args.compute_voxel_grid_on_cpu:
                            event_tensor = events_to_voxel_grid(event_window,
                                                                num_bins=model.num_bins,
                                                                width=width,
                                                                height=height)
                            event_tensor = torch.from_numpy(event_tensor)
                        else:
                            event_tensor = events_to_voxel_grid_pytorch(event_window,
                                                                        num_bins=model.num_bins,
                                                                        width=width,
                                                                        height=height,
                                                                        device=device)
                            # print(event_tensor.numpy())
                    # print(event_tensor.min())
                    num_events_in_window = num_pol_event
                    reconstructor.update_reconstruction(event_tensor, start_index + num_events_in_window, last_timestamp)

                    start_index += num_events_in_window

                
                # cv2.waitKey(1)

        except KeyboardInterrupt:
            
            device1.shutdown()
            cv2.destroyAllWindows()
            break


    # if args.compute_voxel_grid_on_cpu:
    #     print('Will compute voxel grid on CPU.')

    # # if args.fixed_duration:
    # #     event_window_iterator = FixedDurationEventReader(path_to_events,
    # #                                                      duration_ms=args.window_duration,
    # #                                                      start_index=start_index)
    # # else:
    # #     event_window_iterator = FixedSizeEventReader(path_to_events, num_events=N, start_index=start_index)

    # with Timer('Processing entire dataset'):
    #     for event_window in event_window_iterator:
    #         with Timer('Test'):
    #             last_timestamp = event_window[-1, 0]

    #             with Timer('Building event tensor'):
    #                 if args.compute_voxel_grid_on_cpu:
    #                     event_tensor = events_to_voxel_grid(event_window,
    #                                                         num_bins=model.num_bins,
    #                                                         width=width,
    #                                                         height=height)
    #                     event_tensor = torch.from_numpy(event_tensor)
    #                 else:
    #                     event_tensor = events_to_voxel_grid_pytorch(event_window,
    #                                                                 num_bins=model.num_bins,
    #                                                                 width=width,
    #                                                                 height=height,
    #                                                                 device=device)

    #             num_events_in_window = event_window.shape[0]
    #             reconstructor.update_reconstruction(event_tensor, start_index + num_events_in_window, last_timestamp)

    #             start_index += num_events_in_window
