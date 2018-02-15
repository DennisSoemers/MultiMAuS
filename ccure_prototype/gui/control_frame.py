"""
This module implements a Control UI which can be used to control C-Cure prototype experiments (pause/resume, quit)

@author Dennis Soemers
"""

from tkinter import *

control_frame_alive = False


def create_control_frame():
    global control_frame_alive

    root = Tk()
    control_ui = ControlFrame(root)
    control_frame_alive = True

    root.protocol("WM_DELETE_WINDOW", on_frame_exit)

    return control_ui


def is_control_frame_alive():
    global control_frame_alive
    return control_frame_alive


def on_frame_exit():
    global control_frame_alive
    control_frame_alive = False


class ControlFrame(Frame):

    def __init__(self, root):
        super().__init__(root)

        self.want_pause = False
        self.want_quit = False

        self.root = root
        self.root.title("C-Cure")
        self.root.geometry("800x400")

        self.status_label = Label(root, text="Current Status: ")
        self.status_label.pack(fill="both", expand=1)

        self.timesteps_info_label = Label(
            root,
            text="Finished timesteps: {0} / {1} ({2:.2f}%, speed = {3:.1f} timesteps per second)".format(0, 0, 0, 0))
        self.timesteps_info_label.pack(fill="both", expand=1)

        self.transactions_label = Label(
            root,
            text="Num allowed transactions: {0}. Num genuine: {1} ({2:.2f}%). Num fraudulent: {3} ({4:.2f}%)".format(
                0, 0, 0, 0, 0)
        )
        self.transactions_label.pack(fill="both", expand=1)

        self.authentications_label = Label(
            root,
            text="Num secondary authentications: {0} ({1:.2f}%). "
                 "Num genuine: {2} ({3:.2f}% of all genuine transactions). "
                 "Num fraudulent: {4} ({5:.2f}% of all fraudulent transactions)".format(
                     0, 0, 0, 0, 0, 0)
        )
        self.authentications_label.pack(fill="both", expand=1)

        self.pause_button = Button(root, text="Pause", command=self.pause)
        self.pause_button.pack(fill="both", expand=1)

        self.quit_button = Button(root, text="Quit", command=self.quit)
        self.quit_button.pack(fill="both", expand=1)

    def pause(self):
        self.want_pause = not self.want_pause

        if self.want_pause:
            self.pause_button["text"] = "Resume"
        else:
            self.pause_button["text"] = "Pause"

    def set_status(self, status):
        self.status_label["text"] = "Current Status: " + status

    def quit(self):
        self.want_quit = True

    def update_info_labels(self, finished_timesteps, goal_timesteps, timestep_speed,
                           num_transactions=0, num_genuines=0, num_fraudulents=0,
                           num_secondary_auths=0, num_secondary_auths_genuine=0, num_secondary_auths_fraud=0):

        self.timesteps_info_label["text"] = \
            "Finished timesteps: {0} / {1} ({2:.2f}%, speed = {3:.1f} timesteps per second)".format(
                finished_timesteps, goal_timesteps,
                (float(finished_timesteps) / goal_timesteps) * 100, timestep_speed)

        if num_transactions > 0:
            self.transactions_label["text"] = \
                "Num transactions: {0}. Num genuine: {1} ({2:.2f}%). Num fraudulent: {3} ({4:.2f}%)".format(
                    num_transactions,
                    num_genuines, (100.0 * num_genuines) / num_transactions,
                    num_fraudulents, (100.0 * num_fraudulents) / num_transactions)

        # avoid divisions by 0
        if num_transactions == 0:
            num_transactions = 1
        if num_genuines == 0:
            num_genuines = 1
        if num_fraudulents == 0:
            num_fraudulents = 1

        if num_secondary_auths > 0:
            self.authentications_label["text"] = \
                "Num secondary authentications: {0} ({1:.2f}%). " \
                "Num genuine: {2} ({3:.2f}% of all genuine transactions). " \
                "Num fraudulent: {4} ({5:.2f}% of all fraudulent transactions)".format(
                    num_secondary_auths,
                    (100.0 * num_secondary_auths) / num_transactions,
                    num_secondary_auths_genuine,
                    (100.0 * num_secondary_auths_genuine) / num_genuines,
                    num_secondary_auths_fraud,
                    (100.0 * num_secondary_auths_fraud) / num_fraudulents)
