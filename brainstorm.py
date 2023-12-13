import tkinter as tk

# Funktion, die aufgerufen wird, wenn der Button geklickt wird
def button_click():
    label.config(text="Hallo")

# Erstellen des Hauptfensters
root = tk.Tk()
root.title("Button Beispiel")

# Erstellen eines Labels, das den Text anzeigt
label = tk.Label(root, text="")
label.pack()

# Erstellen eines Buttons
button = tk.Button(root, text="Klick mich", command=button_click)
button.pack()

# Starten der Tkinter-Schleife
root.mainloop()

0100000101110000
0110000101100011
0110100001100101
