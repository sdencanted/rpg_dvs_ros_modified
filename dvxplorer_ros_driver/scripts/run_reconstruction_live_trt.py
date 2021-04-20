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
from image_reconstructor_trt import ImageReconstructor
from options.inference_options import set_inference_options
# import onnx
# import onnx_tensorrt.backend_1 as backend
import numpy as np
# from torch2trt import torch2trt
from torch2trt import TRTModule




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

    width,height = 320,240
    print('Sensor size: {} x {}'.format(width, height))

    # Load model
    
    model_trt = TRTModule()
    model_trt.load_state_dict(torch.load(args.path_to_model))
    device = get_device(args.use_gpu)
    num_bins = 5

    # model.eval()

    reconstructor = ImageReconstructor(model_trt, height, width, num_bins, args)

    """ Read chunks of events using Pandas """

    # Loop through the events and reconstruct images
    N = args.window_size
    if not args.fixed_duration:
        N = int(width * height * args.num_events_per_pixel)
        print('Will use {} events per tensor (automatically estimated with num_events_per_pixel={:0.2f}).'.format(N, args.num_events_per_pixel))

    initial_offset = args.skipevents
    sub_offset = args.suboffset
    start_index = initial_offset + sub_offset
    print(initial_offset,sub_offset,start_index)



    # device1 = DAVIS(noise_filter=False)
    device1 = DVXPLORER()
    device1.start_data_stream()
    # load new config
    device1.set_bias_from_json("./scripts/configs/dvxplorer_config.json")
    
    while True:
        try:
            (pol_events, num_pol_event,
            special_events, num_special_event,
            imu_events, num_imu_event) = \
                device1.get_event("events")

            if num_pol_event != 0:

                
                event_window = pol_events.astype(np.float32)
                event_window[:,0] = event_window[:,0]/1e6
                # print(event_window)



                with Timer('Processing entire dataset'):
                    last_timestamp = event_window[-1, 0]
                    # print(event_window[0, 0],event_window[-1, 0])


                    with Timer('Building event tensor'):
                        if args.compute_voxel_grid_on_cpu:
                            event_tensor = events_to_voxel_grid(event_window,
                                                                num_bins=num_bins,
                                                                width=width,
                                                                height=height)
                            event_tensor = torch.from_numpy(event_tensor)
                        
                            # print('run 1')
                        else:
                            event_tensor = events_to_voxel_grid_pytorch(event_window,
                                                                        num_bins=num_bins,
                                                                        width=width,
                                                                        height=height,
                                                                        device=device)
                           
                    num_events_in_window = num_pol_event
                    
                    reconstructor.update_reconstruction(event_tensor, start_index + num_events_in_window, last_timestamp)

                    start_index += num_events_in_window

                
                # cv2.waitKey(1)

        except KeyboardInterrupt:
            
            device1.shutdown()
            cv2.destroyAllWindows()
            break

