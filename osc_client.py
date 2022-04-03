import argparse
import random
import time
from pythonosc import udp_client

class MonoBehaviour():

    def __init__(self,ip,port):
        self.DISABLE = 0
        self.ENABLE_TRACKER = 1
        self.ENABLE_CONTROLLER_L = 2
        self.ENABLE_CONTROLLER_R = 3
        self.ENABLE_TRACKING_REFERENCE = 4

        parser = argparse.ArgumentParser()
        parser.add_argument("--ip", default=ip,
            help="The ip of the OSC server")
        parser.add_argument("--port", type=int, default=port,
            help="The port the OSC server is listening on")
        args = parser.parse_args()

        self.client = udp_client.SimpleUDPClient(args.ip, args.port)

    def send_joint_unity(self,index,enable,timeoffset,serial,**kwargs):
        if kwargs != None:
            pass 
        else:
            message = '/VMT/Joint/Unity'
            arguments=[int(index),int(enable),float(timeoffset),
                float(kwargs["x"]),
                float(kwargs["y"]),
                float(kwargs["z"]),
                float(kwargs["qx"]),
                float(kwargs["qy"]),
                float(kwargs["qz"]),
                float(kwargs["qw"]),
                string(serial),
                        ]
            self.client.send_message(f"{message}", arguments)

    def send_room_unity(self,index,enable,timeoffset,**kwargs):
        if kwargs == None:
            print(f'pass')
            # print(kwargs)
        else:
            message = '/VMT/Room/Unity'
            arguments=[int(index),int(enable),float(timeoffset),
                float(kwargs["x"]),
                float(kwargs["y"]),
                float(kwargs["z"]),
                float(kwargs["qx"]),
                float(kwargs["qy"]),
                float(kwargs["qz"]),
                float(kwargs["qw"]),
                        ]
            self.client.send_message(f"{message}", arguments)
            # print(f'send text {message} {arguments}')
    
    def update(self):
        self.send_joint_unity(0, self.ENABLE_TRACKER,       head,       "VMT_10")
        self.send_joint_unity(1, self.ENABLE_CONTROLLER_L,  leftHand,   "VMT_10")
        self.send_joint_unity(2, self.ENABLE_CONTROLLER_R,  rightHand,  "VMT_10")
        self.send_joint_unity(3, self.ENABLE_TRACKER,       waist,      "VMT_10")
        self.send_joint_unity(4, self.ENABLE_TRACKER,       leftFoot,   "VMT_10")
        self.send_joint_unity(5, self.ENABLE_TRACKER,       rightFoot,  "VMT_10")

        self.send_room_unity(10,self.ENABLE_TRACKING_REFERENCE, zeroReference)


if __name__ == "__main__":
    ip="127.0.0.1"
    port="5005"
    # port="39570"
    monobehaviour=MonoBehaviour(ip,port)

    # monobehaviour.send_room_unity()
    # monobehaviour.send_joint_unity()
    monobehaviour.update()
    time.sleep(1)