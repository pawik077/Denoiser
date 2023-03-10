# import tkinter as tk


# class GUI(tk.Tk):
#     def __init__(self):
#         super().__init__()
#         self.title("GUI")
#         self.geometry("500x500")
#         self.resizable(False, False)
#         self.config(bg="#000000")
#         self.create_widgets()

#     def create_widgets(self):
#         self.btn = tk.Button(self, text="Button", command=self.btn_click)
#         self.btn.pack()

#     def btn_click(self):
#         print("Button clicked!")

# if __name__ == "__main__":
#     gui = GUI()
#     gui.mainloop()

# from tkinter import *
# from tkinter import ttk
# root = Tk()
# frm = ttk.Frame(root, padding=10)
# frm.grid()
# ttk.Label(frm, text="Hello World!").grid(column=0, row=0)
# ttk.Button(frm, text="Quit", command=root.destroy).grid(column=1, row=0)
# root.mainloop()
import tkinter as tk
import tkinter.ttk as ttk
from train import train

# import augmentation

models = ["REDNet", "MWCNN", "PRIDNet"]


class GUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("GUI")
        self.geometry("500x500")
        #self.resizable(False, False)
        #self.config(bg="#000000")
        self.datasets = {
            "SIDD": tk.BooleanVar(value=True),
            "RENOIR": tk.BooleanVar(value=True),
            "NIND": tk.BooleanVar(value=True)
        }
        self.augmentations = {
            "Up-down flip": tk.BooleanVar(value=True),
            "Left-right flip": tk.BooleanVar(value=True),
            "Rotate": tk.BooleanVar(value=True),
            "Adjust hue": tk.BooleanVar(value=False),
            "Adjust brightness": tk.BooleanVar(value=False),
            "Adjust contrast": tk.BooleanVar(value=False),
            "Adjust saturation": tk.BooleanVar(value=False)
        }
        self.create_widgets()

    def create_widgets(self):
        self.btn = ttk.Button(self, text="Debug", command=self.debug)
        self.btn.pack()
        
        self.datasets_label = ttk.Label(self, text="Datasets:")
        self.datasets_label.pack()
        self.datasets_cbs = [tk.Checkbutton(self, text=k, variable=v, onvalue=True, offvalue=False) for (k,v) in self.datasets.items()]
        for _, cb in enumerate(self.datasets_cbs): cb.pack()

        self.model_label = ttk.Label(self, text="Model:")
        self.model_label.pack()
        self.model = ttk.Combobox(self, values=models, state="readonly")
        self.model.bind("<<ComboboxSelected>>", self.change_model)
        self.model.pack()
        self.model.current(0)

        self.augment_label = ttk.Label(self, text="Augmentations:")
        self.augment_label.pack()
        self.augment_cbs = [tk.Checkbutton(self, text=k, variable=v, onvalue=True, offvalue=False) for (k,v) in self.augmentations.items()]
        for _, cb in enumerate(self.augment_cbs): cb.pack()

        self.filename_label = ttk.Label(self, text="Filename:")
        self.filename_label.pack()
        self.filename = ttk.Entry(self)
        self.filename.insert(0, self.model.get() + ".h5")
        self.filename.pack()

        self.epochs_label = ttk.Label(self, text="Epochs:")
        self.epochs_label.pack()
        self.epochs = ttk.Entry(self)
        self.epochs.insert(0, 200)
        self.epochs.pack()



        self.train_btn = ttk.Button(self, text="Train", command=self.train_trigger)
        self.train_btn.pack()

    def train_trigger(self):
        datasets = [k for k,v in self.datasets.items() if v.get()]
        model = self.model.get()
        augmentations = [k for k,v in self.augmentations.items() if v.get()]
        filename = self.filename.get()
        epochs = int(self.epochs.get())
        #train(datasets, model, augmentations, filename, epochs)
        pass

    def change_model(self, *args):
        self.filename.delete(0, tk.END)
        self.filename.insert(0, self.model.get() + ".h5")
    
    def debug(self):
        print(f"Datasets: {[k for k,v in self.datasets.items() if v.get()]}")
        print(f"Model: {self.model.get()}")
        print(f'Augmentations: {[k for k,v in self.augmentations.items() if v.get()]}')
        print(f"Filename: {self.filename.get()}")
        print(f"Epochs: {self.epochs.get()}")


if __name__ == "__main__":
    gui = GUI()
    gui.mainloop()