import tkinter as tk
from tkinter import ttk

from PIL import ImageTk

from RobotPipelines import LeaderPipeline, FollowerPipeline
from RobotModel import RobotModel


class Application(tk.Frame):

    fps = 10
    ms = int(1000/fps)
    title = "Proyecto de Percepción y Manipulación"
    leader_model = RobotModel.GIRONA_500_1
    follower_model = RobotModel.GIRONA_500_2

    def __init__(self, master):
        super().__init__(master)
        self.master = master
        self.master.title(Application.title)
        self.pack()
        self.master.resizable(width=True, height=True)
        self.leader_pipeline = None
        self.follower_pipeline = None
        self._create_widgets()

    def _create_widgets(self):
        self.quit = tk.Button(self, text="QUIT", fg="red",
                              command=self._quit)
        self.quit.pack(side=tk.BOTTOM, pady=5)

        self.ip_field = ttk.Entry(self)
        self.ip_field.insert(0, "127.0.0.1")
        self.ip_field.pack(side=tk.TOP, pady=5)

        self.image_canvas = tk.Canvas(self)
        self.image_canvas.configure(bg="gray")
        self.image_canvas.pack(side=tk.LEFT)

        self.image_canvas2 = tk.Canvas(self)
        self.image_canvas2.configure(bg="gray")
        self.image_canvas2.pack(side=tk.RIGHT)

        self.start_button = ttk.Button(self)
        self.start_button["text"] = "Start"
        self.start_button["command"] = self._start_streaming
        self.start_button.pack(side=tk.TOP, pady=5)

    def _start_streaming(self):
        # Start streaming task
        if self.leader_pipeline:
            self.leader_pipeline.stop()
        if self.follower_pipeline:
            self.follower_pipeline.stop()

        self.leader_pipeline = LeaderPipeline(
            address=self.ip_field.get(),
            robot_model=Application.leader_model,
            adq_rate=Application.fps)
        self.follower_pipeline = LeaderPipeline(
            address=self.ip_field.get(),
            robot_model=Application.follower_model,
            adq_rate=Application.fps)

        self.leader_pipeline.start()
        self.follower_pipeline.start()

        self.task_id = self.after(ms=0, func=self._refresh)

    def _refresh(self):
        frame1 = self.leader_pipeline.get_last_frame()
        if frame1:
            self._frame1 = ImageTk.PhotoImage(image=frame1)  # prevent GC by setting it as an instance field
            self.image_canvas.configure(width=frame1.width, height=frame1.height)
            self.image_canvas.create_image(0, 0, image=self._frame1, anchor=tk.NW)

        frame2 = self.follower_pipeline.get_last_frame()
        if frame2:
            self._frame2 = ImageTk.PhotoImage(image=frame2)  # prevent GC by setting it as an instance field
            self.image_canvas2.configure(width=frame2.width, height=frame2.height)
            self.image_canvas2.create_image(0, 0, image=self._frame2, anchor=tk.NW)

        # Reschedule
        self.after(ms=self.ms, func=self._refresh)

    def _quit(self):
        if self.leader_pipeline:
            self.leader_pipeline.stop()
        if self.follower_pipeline:
            self.follower_pipeline.stop()
        self.master.destroy()


def start():
    root = tk.Tk()
    app = Application(master=root)
    app.mainloop()
