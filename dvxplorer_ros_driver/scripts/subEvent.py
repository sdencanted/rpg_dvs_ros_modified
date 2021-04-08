#!/usr/bin/env python
from __future__ import print_function, absolute_import
import rospy
from dvs_msgs.msg import EventArray
from std_msgs.msg import Int32, Float32MultiArray
from rospy.numpy_msg import numpy_msg

import numpy as np
import cv2
import torch
from utils.loading_utils import load_model, get_device
import argparse
import pandas as pd
from utils.event_readers import FixedSizeEventReader, FixedDurationEventReader
from utils.inference_utils import events_to_voxel_grid, events_to_voxel_grid_pytorch
from utils.timers import Timer
import time
from image_reconstructor_trt import ImageReconstructor
from options.inference_options import set_inference_options

# from torch2trt import torch2trt
from torch2trt import TRTModule

class subEvent:
    def __init__(self,args,reconstructor,width,height,num_bins,device):
        self.reconstructor = reconstructor
        self.args = args
        self.width = width
        self.height = height
        self.device = device
        self.num_bins = num_bins
        self.initial_offset = self.args.skipevents
        self.sub_offset = self.args.suboffset
        self.start_index = self.initial_offset + self.sub_offset

        print(self.initial_offset,self.sub_offset,self.start_index)

        self.N = self.args.window_size
        if not self.args.fixed_duration:
            self.N = int(self.width * self.height * self.args.num_events_per_pixel)
            print('Will use {} events per tensor (automatically estimated with num_events_per_pixel={:0.2f}).'.format(self.N, self.args.num_events_per_pixel))

        ## Initialize node and subscriber

        self.event_sub = rospy.Subscriber('/dvs/eventsArr', Float32MultiArray,self.eventsArrCallback)


        
    
    def eventsArrCallback(self,data):
        with Timer('Events -> Device (voxel grid)'):
            event_window = torch.Tensor(data.data).view(-1,4).to(self.device)
            print(event_window.shape)


        # try:
        #     with Timer('Processing entire dataset'):
        #         last_timestamp = event_window[-1, 0]


        #         with Timer('Building event tensor'):
        #             if self.args.compute_voxel_grid_on_cpu:
        #                 event_tensor = events_to_voxel_grid(event_window,
        #                                                     num_bins=self.num_bins,
        #                                                     width=self.width,
        #                                                     height=self.height)
        #                 event_tensor = torch.from_numpy(event_tensor)

        #             else:
        #                 event_tensor = events_to_voxel_grid_pytorch(event_window,
        #                                                             num_bins=self.num_bins,
        #                                                             width=self.width,
        #                                                             height=self.height,
        #                                                             device=self.device)
                        
        #         num_events_in_window = event_window.size(0)
        #         print(num_events_in_window)
        #         reconstructor.update_reconstruction(event_tensor, self.start_index + num_events_in_window, last_timestamp)

        #         self.start_index += num_events_in_window
                
        # except KeyboardInterrupt:
            
        #         device1.shutdown()
        #         cv2.destroyAllWindows()
        #         break
    

# def eventsCallback(data):
#     # time_init = data.events[0].ts.nsecs
#     # for event in data.events:
#         # rospy.loginfo('x: %i, y: %i, pol: %i',event.x,event.y,event.polarity)
#         # rospy.loginfo('y: %i',event.y)
#         # rospy.loginfo('ts: %i',event.ts.secs)
#         # rospy.loginfo('pol %i:',event.polarity)
#     rospy.loginfo(data.events)
#     # rospy.loginfo(rospy.get_caller_id() + 'I heard ', data.events)


# def eventsSizeCallback(data):
#     rospy.loginfo('Event size: %i', data.data)


if __name__ == '__main__':
    print('run main')
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

    rospy.init_node('subEvents',anonymous=True)

    subE = subEvent(args,reconstructor,width,height,num_bins,device)
    rospy.spin()