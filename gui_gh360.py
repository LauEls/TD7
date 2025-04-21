import os
import re
import sys
import TKinterModernThemes as TKMT
import tkinter as tk
from threading import Event, Thread

from gh360_class import RL_GH360
from dummy_class import Dummy

class GH360LearningGUI(TKMT.ThemedTKinterFrame):
    def __init__(self):
        super().__init__("Learning Experiments", "sun-valley", "dark")
        
        self.Text("Environment: ", row=0, col=0)
        self.env_option_menu_list = ["Door"]
        self.env_name = tk.StringVar(value=self.env_option_menu_list[0])
        self.OptionMenu(self.env_option_menu_list, self.env_name, row=0, col=1)
        self.env_name.trace_add("write", self.options_changed)

        self.Text("Controller: ", row=1, col=0)
        self.controller_option_menu_list = ["EEF Velocity", "Motor Velocity"]
        self.controller_name = tk.StringVar(value=self.controller_option_menu_list[0])
        self.OptionMenu(self.controller_option_menu_list, self.controller_name, row=1, col=1)
        self.controller_name.trace_add("write", self.options_changed)
        
        self.Text("Learning Mode: ", row=2, col=0)
        self.learning_mode_option_menu_list = ["online", "offline"]
        self.learning_mode_name = tk.StringVar(value=self.learning_mode_option_menu_list[0])
        self.OptionMenu(self.learning_mode_option_menu_list, self.learning_mode_name, row=2, col=1)
        self.learning_mode_name.trace_add("write", self.options_changed)

        self.Text("Config Folder: ", row=3, col=0)
        self.config_folder_option_menu_list = self.parse_config_folders()
        self.config_folder_name = tk.StringVar(value=self.config_folder_option_menu_list[0])
        self.opmen_config_folder = self.OptionMenu(self.config_folder_option_menu_list, self.config_folder_name, row=3, col=1)

        # self.Text("Experimental Runs: ", row=4, col=0)
        # self.experimental_runs = tk.IntVar(value=1)
        # exp_runs_obj = self.NumericalSpinbox(1,10,1,self.experimental_runs, row=4, col=1)
        # exp_runs_obj.state(["readonly"])
        

        self.btn_load_config = self.AccentButton("Load Config", row=5, col=0, colspan=2, command=self.load_config)
        self.btn_start = self.AccentButton("Start", row=6, col=0, command=self.start_experiment)
        self.btn_start.state(["disabled"])
        self.btn_stop = self.AccentButton("Stop", row=6, col=1, command=self.stop_experiment)
        self.btn_stop.state(["disabled"])

        self.stop_learning = Event()

        self.run()

    def options_changed(self, *args):
        self.config_folder_option_menu_list = self.parse_config_folders()
        self.config_folder_name = tk.StringVar(value=self.config_folder_option_menu_list[0])
        self.opmen_config_folder.set_menu(self.config_folder_name.get(), *self.config_folder_option_menu_list)

    def parse_config_folders(self):
        config_folders = []

        config_path = "runs/"
        #Environment
        if self.env_name.get() == "Door":
            config_path += "door/"
        
        config_path += "real_gh360/"

        if self.controller_name.get() == "EEF Velocity":
            config_path += "eef_vel/"
        elif self.controller_name.get() == "Motor Velocity":
            config_path += "motor_vel/"

        config_path += self.learning_mode_name.get() + "/"

        # dir_lvl = 0
        for root, dirs, files in os.walk(config_path):
            for dir in dirs:
                config_folders.append(dir)
                print("Found config folder:", dir)
            break

        self.config_path = config_path
        return config_folders
    
    def load_config(self):
        # Placeholder for loading config
        print("Loading config...")
        self.btn_load_config.state(["disabled"])

        self.learning_class = RL_GH360(self.config_path + self.config_folder_name.get())
        # self.learning_class = Dummy()

        self.btn_start.state(["!disabled"])

    def start_experiment(self):
        # Placeholder for starting experiment
        print("Starting experiment...")
        self.btn_start.state(["disabled"])
        self.learning_thread = Thread(target=self.learning_class.start_learning, args=(self.stop_learning,))
        self.learning_thread.start()
        self.btn_stop.state(["!disabled"])

    def stop_experiment(self):
        # Placeholder for stopping experiment
        print("Stopping experiment...")
        self.btn_stop.state(["disabled"])
        self.stop_learning.set()
        self.learning_thread.join()
        self.btn_start.state(["!disabled"])

GH360LearningGUI()