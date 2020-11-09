import tkinter as tk
from tkinter import ttk
from tkinter import messagebox

from PIL import ImageTk

from ImagePipeline import ImagePipeline


class Application(tk.Frame):

    fps = 10
    ms = int(1000/fps)
    title = "Proyecto de Percepción y Manipulación"

    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.master.title(Application.title)
        self.pack()
        self.master.resizable(width=True, height=True)
        self.pipeline = None
        self._create_widgets()

    def _create_widgets(self):
        self.quit = tk.Button(self, text="QUIT", fg="red",
                              command=self._quit)
        self.quit.pack(side=tk.BOTTOM, pady=5)

        self.combobox = ttk.Combobox(self, state="readonly")
        self.combobox["values"] = ["Girona 500 1", "Girona 500 2"]
        self.combobox.current(0)
        self.combobox.pack(side=tk.TOP, pady=5)

        self.ip_field = ttk.Entry(self)
        self.ip_field.insert(0, "127.0.0.1")
        self.ip_field.pack(side=tk.TOP, pady=5)

        self.image_canvas = tk.Canvas(self)
        self.image_canvas.configure(bg='gray')
        self.image_canvas.pack(side=tk.BOTTOM)

        self.start_button = ttk.Button(self)
        self.start_button["text"] = "Start"
        self.start_button["command"] = self._start_streaming
        self.start_button.pack(side=tk.TOP, pady=5)

    def _start_streaming(self):
        index = self.combobox.current()
        if index >= 0:
            # Start streaming task
            if self.pipeline:
                self.pipeline.stop()
            self.pipeline = ImagePipeline(
                address=self.ip_field.get(),
                port=self.get_port(index+1),
                adq_rate=Application.fps)
            self.pipeline.start()
            self.task_id = self.image_canvas.after(ms=0, func=self._load_image)
        else:
            messagebox.showwarning(title="No Girona robot selected",
                                   message="Please select an item from the list")

    @staticmethod
    def get_port(girona_index):
        if girona_index == 1:
            return 8000
        elif girona_index == 2:
            return 8010
        else:
            raise ValueError("Invalid Girona 500 index: " + str(girona_index))

    def _load_image(self):
        frame = self.pipeline.get_last_frame()
        if frame:
            self._frame = ImageTk.PhotoImage(image=frame)  # prevent GC by setting it as an instance field
            self.image_canvas.configure(width=frame.size[0], height=frame.size[1])
            self.image_canvas.create_image(0, 0, image=self._frame, anchor=tk.NW)

        # Reschedule
        self.image_canvas.after(ms=self.ms, func=self._load_image)

    def _quit(self):
        if self.pipeline:
            self.pipeline.stop()
        self.master.destroy()


def start():
    root = tk.Tk()
    app = Application(master=root)
    app.mainloop()
