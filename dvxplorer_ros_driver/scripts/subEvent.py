#!/usr/bin/env python

import rospy
from dvs_msgs.msg import EventArray
from std_msgs.msg import Int32, Float32MultiArray
from rospy.numpy_msg import numpy_msg
import numpy as np

def eventsCallback(data):
    # time_init = data.events[0].ts.nsecs
    # for event in data.events:
        # rospy.loginfo('x: %i, y: %i, pol: %i',event.x,event.y,event.polarity)
        # rospy.loginfo('y: %i',event.y)
        # rospy.loginfo('ts: %i',event.ts.secs)
        # rospy.loginfo('pol %i:',event.polarity)
    rospy.loginfo(data.events)
    # rospy.loginfo(rospy.get_caller_id() + 'I heard ', data.events)

def eventsArrCallback(data):
    # time_init = data.events[0].ts.nsecs
    # for event in data.events:
        # rospy.loginfo('x: %i, y: %i, pol: %i',event.x,event.y,event.polarity)
        # rospy.loginfo('y: %i',event.y)
        # rospy.loginfo('ts: %i',event.ts.secs)
        # rospy.loginfo('pol %i:',event.polarity)
    event = np.array(data.data)
    event = event.reshape(-1,4)
    rospy.loginfo(event.shape)
    # rospy.loginfo(rospy.get_caller_id() + 'I heard ', data.events)


def eventsSizeCallback(data):
    
    rospy.loginfo('Event size: %i', data.data)

def subEvent():


    rospy.init_node('subEvent', anonymous=True)
    #rospy.Subscriber('/dvs/events', EventArray,eventsCallback)
    rospy.Subscriber('/dvs/eventsArr', Float32MultiArray,eventsArrCallback)
    rospy.Subscriber('/dvs/events_size', Int32, eventsSizeCallback)



    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    print('run main')
    subEvent()
