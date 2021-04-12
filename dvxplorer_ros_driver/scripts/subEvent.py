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
        self.args = args
        self.reconstructor = reconstructor
        self.width = width
        self.height = height
        self.num_bins = num_bins
        self.device = device
        self.initial_offset = self.args.skipevents
        self.sub_offset = self.args.suboffset
        self.start_index = self.initial_offset + self.sub_offset
        # self.event_window = None

        print(self.initial_offset,self.sub_offset,self.start_index)


        ## Initialize node and subscriber
        self.event_sub = rospy.Subscriber('/dvs/eventsArr', Float32MultiArray,self.eventsArrCallback)


        
    
    def eventsArrCallback(self,data):
        if data.data is not None:
            event_window = torch.Tensor(data.data).view(-1,4).to(self.device)
        
        with Timer('Processing entire dataset'):

            last_timestamp = event_window[-1, 0]
            print(event_window[0,0],event_window[-1,0])

            with Timer('Building event tensor'):
                if args.compute_voxel_grid_on_cpu:
                    event_tensor = events_to_voxel_grid(event_window,
                                                        num_bins=self.num_bins,
                                                        width=self.width,
                                                        height=self.height)
                    event_tensor = torch.from_numpy(event_tensor)
                

                else:
                    event_tensor = events_to_voxel_grid_pytorch(event_window,
                                                                num_bins=self.num_bins,
                                                                width=self.width,
                                                                height=self.height,
                                                                device=self.device)
                    
            num_events_in_window = event_window.shape[0]
            self.reconstructor.update_reconstruction(event_tensor, self.start_index + num_events_in_window, last_timestamp)

            self.start_index += num_events_in_window
            with Timer('Show img'):
                self.show_image(self.reconstructor.out)

    def show_image(self,img):
      cv2.imshow("Image Window", img)
      cv2.waitKey(1)

            


def run(args,reconstructor,width,height,num_bins,device):
    # Initialize node
    rospy.init_node('subEvents',anonymous=True)


    #Intilize callback class
    subE = subEvent(args,reconstructor,width,height,num_bins,device)
    while not rospy.is_shutdown():
        rospy.spin()

    # while True:
    #     with Timer('Processing entire dataset'):

    #         last_timestamp = subE.event_window[-1, 0]

    #         with Timer('Building event tensor'):
    #             if args.compute_voxel_grid_on_cpu:
    #                 event_tensor = events_to_voxel_grid(subE.event_window,
    #                                                     num_bins=num_bins,
    #                                                     width=width,
    #                                                     height=height)
    #                 event_tensor = torch.from_numpy(event_tensor)
                

    #             else:
    #                 event_tensor = events_to_voxel_grid_pytorch(subE.event_window,
    #                                                             num_bins=num_bins,
    #                                                             width=width,
    #                                                             height=height,
    #                                                             device=device)
                    
    #         num_events_in_window = subE.event_window.shape[0]
    #         reconstructor.update_reconstruction(event_tensor, subE.start_index + num_events_in_window, last_timestamp)

    #         subE.start_index += num_events_in_window

        # try:
        #     rospy.spin()
        # except KeyboardInterrupt:
        #     print('Shutting down...')

        # cv2.destroyAllWindows()

            


if __name__ == '__main__':
    print('run main')
    parser = argparse.ArgumentParser(
        description='Evaluating a trained network')
    parser.add_argument('-c', '--path_to_model', required=True, type=str,
                        help='path to model weights')
    parser.add_argument('--fixed_duration', dest='fixed_duration', action='store_true')
    parser.set_defaults(fixed_duration=False)
    parser.add_argument('-N', '--window_size', default=None, type=int,
                        help="Size of each event window, in number of events. Ignored if --fixed_duration=True")
    parser.add_argument('--num_events_per_pixel', default=0.35, type=float,
                        help='in case N (window size) is not specified, it will be \
                              automatically computed as N = width * height * num_events_per_pixel')
    parser.add_argument('--skipevents', default=0, type=int)
    parser.add_argument('--suboffset', default=0, type=int)
    parser.add_argument('--compute_voxel_grid_on_cpu', dest='compute_voxel_grid_on_cpu', action='store_true')
    parser.set_defaults(compute_voxel_grid_on_cpu=False)

    # Set inference options
    set_inference_options(parser)

    # Converts the args into dict format
    args = parser.parse_args()

    # Initialize camera model (image heigh and width) [DVXplorer Lite: 240x320]
    width,height = 320,240
    print('Sensor size: {} x {}'.format(width, height))

    # Load model
    model_trt = TRTModule()
    model_trt.load_state_dict(torch.load(args.path_to_model))
    device = get_device(args.use_gpu)
    num_bins = 5

    # model.eval()

    # Image reconstructor
    reconstructor = ImageReconstructor(model_trt, height, width, num_bins, args)

    # Run
    # run(args,reconstructor,width,height,num_bins,device)
    rospy.init_node('subEvents',anonymous=True)


    #Intilize callback class
    subE = subEvent(args,reconstructor,width,height,num_bins,device)

    while not rospy.is_shutdown():
        rospy.spin()

 