import tkinter as tk
import tkinter.ttk as ttk
from train import train
from REDNet import REDNet_model
import augmentation

models = ["REDNet", "MWCNN", "PRIDNet"]


class GUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("GUI")
        self.geometry("500x500")
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
        
        self.frm_datasets = tk.Frame(self)
        self.frm_datasets.pack()
        self.lbl_datasets = ttk.Label(self.frm_datasets, text="Datasets:")
        self.lbl_datasets.pack()
        self.cbs_datasets = [tk.Checkbutton(self.frm_datasets, text=k, variable=v, onvalue=True, offvalue=False) for (k,v) in self.datasets.items()]
        for _, cb in enumerate(self.cbs_datasets): cb.pack()

        self.frm_model = tk.Frame(self)
        self.frm_model.pack()

        self.lbl_model = ttk.Label(self.frm_model, text="Model:")
        self.lbl_model.pack()
        self.cbb_model = ttk.Combobox(self.frm_model, values=models, state="readonly")
        self.cbb_model.bind("<<ComboboxSelected>>", self.change_model)
        self.cbb_model.pack()
        self.cbb_model.current(0)

        self.frm_augment = tk.Frame(self)
        self.frm_augment.pack()

        self.lbl_augment = ttk.Label(self.frm_augment, text="Augmentations:")
        self.lbl_augment.pack()
        self.cbs_augment = [tk.Checkbutton(self.frm_augment, text=k, variable=v, onvalue=True, offvalue=False) for (k,v) in self.augmentations.items()]
        for _, cb in enumerate(self.cbs_augment): cb.pack()

        self.frm_filename = tk.Frame(self)
        self.frm_filename.pack()

        self.lbl_filename = ttk.Label(self.frm_filename, text="Filename:")
        self.lbl_filename.pack()
        self.ent_filename = ttk.Entry(self.frm_filename)
        self.ent_filename.insert(0, self.cbb_model.get() + ".h5")
        self.ent_filename.pack()

        self.frm_epochs = tk.Frame(self)
        self.frm_epochs.pack()

        self.lbl_epochs = ttk.Label(self.frm_epochs, text="Epochs:")
        self.lbl_epochs.pack()
        self.ent_epochs = ttk.Entry(self.frm_epochs)
        self.ent_epochs.insert(0, 200)
        self.ent_epochs.pack()

        self.btn_train = ttk.Button(self, text="Train", command=self.train_trigger)
        self.btn_train.pack()
        
        self.lbl_processing = ttk.Label(self, text='')
        self.lbl_processing.pack()

    def train_trigger(self):
        self.lbl_processing.config(text="Training model (consult terminal)...")
        self.update()
        datasets = [k for k,v in self.datasets.items() if v.get()]
        filename = self.ent_filename.get()
        epochs = int(self.ent_epochs.get())
        match self.cbb_model.get():
            case "REDNet":
                model = REDNet_model()
            case "MWCNN":
                # model = MWCNN_model()
                raise NotImplementedError("MWCNN is not implemented yet")
            case "PRIDNet":
                # model = PRIDNet_model()
                raise NotImplementedError("PRIDNet is not implemented yet")
        augmentations = []
        for a in [k for k,v in self.augmentations.items() if v.get()]:
            match a:
                case "Up-down flip":
                    augmentations.append(augmentation.up_down_flip)
                case "Left-right flip":
                    augmentations.append(augmentation.left_right_flip)
                case "Rotate":
                    augmentations.append(augmentation.rotate)
                case "Adjust hue":
                    augmentations.append(augmentation.adjust_hue)
                case "Adjust brightness":
                    augmentations.append(augmentation.adjust_brightness)
                case "Adjust contrast":
                    augmentations.append(augmentation.adjust_contrast)
                case "Adjust saturation":
                    augmentations.append(augmentation.adjust_saturation)
        train(model, datasets, augmentations, filename, epochs)
        self.lbl_processing.config(text="Done!")
        self.update()

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