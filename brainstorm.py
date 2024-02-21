# import tkinter as tk
#
# # Funktion, die aufgerufen wird, wenn der Button geklickt wird
# def button_click():
#     label.config(text="Hallo")
#
# # Erstellen des Hauptfensters
# root = tk.Tk()
# root.title("Button Beispiel")
#
# # Erstellen eines Labels, das den Text anzeigt
# label = tk.Label(root, text="")
# label.pack()
#
# # Erstellen eines Buttons
# button = tk.Button(root, text="Klick mich", command=button_click)
# button.pack()
#
# # Starten der Tkinter-Schleife
# root.mainloop()

import os
from itunesLibrary import library

path = os.path.join(os.getenv("HOME"), "Music/iTunes/iTunes Music Library.xml")

# must first parse...
lib = library.parse(path)

print(len(lib))    # number of items stored

for playlist in lib.playlists:
    for item in playlist.items:
        print(item)          # perform function on each item in the playlist

# get a single playlist
playlist = lib.getPlaylist("Gray")

# check the playlist type
assert(not playlist.is_smart())
assert(not playlist.is_folder())

# get a list of all of the David Bowie songs
bowie_items = lib.getItemsForArtist("David Bowie")

# get a single song
single_song = lib.getItemsById("16116")

# get the iTunes application version
print(lib.applicationVersion)
