#!/usr/bin/env python

import rospy
from rosgraph_msgs.msg import Clock
from diagnostic_msgs.msg import DiagnosticArray
from dvs_msgs.msg import EventArray
from std_msgs.msg import Int32, Float32MultiArray
class ListenerVilma:
    def __init__(self):
        # self.diagnostics_sub = rospy.Subscriber('diagnostics', DiagnosticArray, self.diagnostics_callback)
        # self.clock_sub = rospy.Subscriber('clock', Clock, self.clock_callback)
        self.eventsize_sub = rospy.Subscriber('/dvs/events_size',Int32,self.eventsSizeCallback)

    def eventsSizeCallback(self,data):
        print(data.data)

    def clock_callback(self, clock):
        print(clos)

if __name__ == '__main__':
    rospy.init_node('listener', anonymous=True)
    myVilma = ListenerVilma()
    rospy.spin()