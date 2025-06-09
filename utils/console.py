import readline
import readchar
import time
import numpy as np

# Config imports
import config.config_quadruped as config_quadruped

class Console():
    def __init__(self, controller_node):
        self.controller_node = controller_node

        # Walking and Stopping
        self.walking = False

        # Go Up and Go Down motion
        self.isDown = True
        self.height_delta = config_quadruped.robot_height

        # Pitch Up and Pitch Down
        self.pitch_delta = 0

        # Step Height holder to keep track of the step height
        self.step_height_holder = config_quadruped.step_height

        # Autocomplete setup
        self.commands = [
            "stw", "ooo", "setStepHeight",
           "goUp", "goDown", "help", "ictp","setupGaitTimer"
        ]
        readline.set_completer(self.complete)
        readline.parse_and_bind("tab: complete")


    def complete(self, text, state):
        options = [cmd for cmd in self.commands if cmd.startswith(text)]
        if state < len(options):
            print(options[state])
            return options[state]
        else:
            return None


    def interactive_command_line(self, ):
        self.print_all_commands()
        while True:
            input_string = input(">>> ")
            try:
                if(input_string == "stw"):
                    if(self.walking):
                        print("The robot is already walking")
                    print("Starting Walking")
                    self.walking = True
                    self.controller_node.mpc.walking = True
                    self.controller_node.mpc.duty_factor = config_quadruped.duty_factor
                elif(input_string == "ooo"):
                    print("Stopping Walking")
                    self.walking = False
                    while(np.sum(self.controller_node.mpc.contact) < 3):
                        time.sleep(0.02)
                    self.controller_node.mpc.duty_factor = 1.0
                    self.controller_node.mpc.contact_time = self.controller_node.mpc.config.timer_t
                    self.controller_node.input[:6] = np.zeros(6)
                    ##TO DO stop walking
                elif(input_string == "goUp"):
                    print("Going Up")
                    start_time = time.time()
                    time_motion = 5.
                    initial_height = self.controller_node.mpc.robot_height
                    delta_height = self.controller_node.mpc.config.robot_height - initial_height
                    while(time.time() - start_time < time_motion):
                        time_diff = time.time() - start_time
                        self.controller_node.mpc.robot_height = initial_height + ( delta_height * time_diff / time_motion)
                        time.sleep(0.01)
                    self.controller_node.isDown = False
                    print("Ready to walk")
                elif(input_string == "goDown"):
                    print("Going Up")
                    start_time = time.time()
                    time_motion = 5.
                    initial_height = self.controller_node.mpc.robot_height
                    delta_height = 0.05 - initial_height
                    print("Initial Height: ", initial_height)
                    while(time.time() - start_time < time_motion):
                        time_diff = time.time() - start_time
                        self.controller_node.mpc.robot_height = initial_height + ( delta_height * time_diff / time_motion)
                        time.sleep(0.01)
                    self.controller_node.isDown = False
                elif(input_string == "setStepHeight"):
                    temp = input("Step Height: >>> ")
                    if(temp != ""):
                        temp = max(0.02, min(float(temp), 0.5))
                        self.controller_node.mpc.step_height = temp
                        
                elif(input_string == "setGaitTimer"):
                    
                    print("Current Step Frequency: ", self.mpc.step_freq)
                    temp = input("Step Frequency: >>> ")
                    if(temp != ""):
                        temp = max(0.4, min(float(temp), 2.0))
                        self.controller_node.mpc.step_freq = temp
                    
                    print("Current Duty Factor: ", self.mpc.duty_factor)
                    temp = input("Duty Factor: >>> ")
                    if(temp != ""):
                        temp = max(0.4, min(float(temp), 0.9))
                        self.controller_node.mpc.duty_factor = temp  

                elif(input_string == "robot_height"):
                    temp = input("Robot Height: >>> ")
                    if(temp != ""):
                        temp = max(0.1, min(float(temp), 0.4))
                        self.controller_node.mpc.robot_height = temp
                
                elif(input_string == "help"):
                    self.print_all_commands()

                
                elif(input_string == "ictp"):
                    print("Interactive Keyboard Control")
                    print("w: Move Forward")
                    print("s: Move Backward")
                    print("a: Move Left")
                    print("d: Move Right")
                    print("q: Rotate Left")
                    print("e: Rotate Right")
                    print("0: Stop")
                    print("1: Pitch Up")
                    print("2: Reset Pitch")
                    print("3: Pitch Down")
                    print("Press any other key to exit")
                    while True:
                        command = readchar.readkey()
                        if(command == "w"):
                            self.controller_node.input[0] += 0.1
                            print("w")
                        elif(command == "s"):
                            self.controller_node.input[0] -= 0.1
                            print("s")
                        elif(command == "a"):
                            self.controller_node.input[1] += 0.05
                            print("a")
                        elif(command == "d"):
                            self.controller_node.input[1] -= 0.05
                            print("d")
                        elif(command == "q"):
                            self.controller_node.input[5] += 0.2
                            print("q")
                        elif(command == "e"):
                            self.controller_node.input[5] -= 0.2
                            print("e")
                        elif(command == "0"):
                            self.controller_node.input[0] = 0
                            self.controller_node.input[1] = 0
                            self.controller_node.input[5] = 0 
                            print("0")
                        # elif(command == "1"):
                        #     self.controller_node.pitch_delta -= 0.1
                        #     print("1")
                        # elif(command == "2"):
                        #     self.controller_node.pitch_delta = 0
                        #     print("2")
                        # elif(command == "3"):
                        #     self.controller_node.pitch_delta += 0.1
                        #     print("3")
                        else:
                            #to do maybe stop the robot
                            break
            except Exception as e:
                print("Error: ", e)
                print("Invalid Command")
                self.print_all_commands()


    def print_all_commands(self):
        print("\nAvailable Commands")
        print("help: Display all available messages")
        print("stw: Start Walking")
        print("ooo: Stop Walking")
        print("ictp: Interactive Keyboard Control")
        print("########################")
        print("narrowstance: Narrow Stance")
        print("widestance: Wide Stance")
        print("goUp: The robot goes up")
        print("goDown: The robot goes down")
        print("########################")
        print("setGaitTimer: Set the gait type")
        print("setupGaitTimer: Setup the gait timer")
        print("setupLegsGains: Setup the leg gains")
        print("setupGeneral: Setup general parameters\n")