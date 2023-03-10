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
        self.btn_debug = ttk.Button(self, text="Debug", command=self.debug)
        self.btn_debug.pack()
        
        self.lbl_datasets = ttk.Label(self, text="Datasets:")
        self.lbl_datasets.pack()
        self.cbs_datasets = [tk.Checkbutton(self, text=k, variable=v, onvalue=True, offvalue=False) for (k,v) in self.datasets.items()]
        for _, cb in enumerate(self.cbs_datasets): cb.pack()

        self.lbl_model = ttk.Label(self, text="Model:")
        self.lbl_model.pack()
        self.cbb_model = ttk.Combobox(self, values=models, state="readonly")
        self.cbb_model.bind("<<ComboboxSelected>>", self.change_model)
        self.cbb_model.pack()
        self.cbb_model.current(0)

        self.lbl_augment = ttk.Label(self, text="Augmentations:")
        self.lbl_augment.pack()
        self.cbs_augment = [tk.Checkbutton(self, text=k, variable=v, onvalue=True, offvalue=False) for (k,v) in self.augmentations.items()]
        for _, cb in enumerate(self.cbs_augment): cb.pack()

        self.lbl_filename = ttk.Label(self, text="Filename:")
        self.lbl_filename.pack()
        self.ent_filename = ttk.Entry(self)
        self.ent_filename.insert(0, self.cbb_model.get() + ".h5")
        self.ent_filename.pack()

        self.lbl_epochs = ttk.Label(self, text="Epochs:")
        self.lbl_epochs.pack()
        self.ent_epochs = ttk.Entry(self)
        self.ent_epochs.insert(0, 200)
        self.ent_epochs.pack()



        self.btn_train = ttk.Button(self, text="Train", command=self.train_trigger)
        self.btn_train.pack()

    def train_trigger(self):
        datasets = [k for k,v in self.datasets.items() if v.get()]
        model = self.cbb_model.get()
        augmentations = [k for k,v in self.augmentations.items() if v.get()]
        filename = self.ent_filename.get()
        epochs = int(self.ent_epochs.get())
        #train(datasets, model, augmentations, filename, epochs)
        pass

    def change_model(self, *args):
        self.ent_filename.delete(0, tk.END)
        self.ent_filename.insert(0, self.cbb_model.get() + ".h5")
    
    def debug(self):
        print(f"Datasets: {[k for k,v in self.datasets.items() if v.get()]}")
        print(f"Model: {self.cbb_model.get()}")
        print(f'Augmentations: {[k for k,v in self.augmentations.items() if v.get()]}')
        print(f"Filename: {self.ent_filename.get()}")
        print(f"Epochs: {self.ent_epochs.get()}")


if __name__ == "__main__":
    gui = GUI()
    gui.mainloop()