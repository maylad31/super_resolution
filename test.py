import PySimpleGUI as sg
sg.theme("DarkTeal2")
layout = [[sg.T("")], [sg.Text("Choose input folder: "), sg.FolderBrowse(key="-IN1-")],[sg.Text("Choose target folder: "), sg.FolderBrowse(key="-IN2-")],[sg.Button("Submit")]]

###Building Window
window = sg.Window('My File Browser', layout, size=(600,150))
    
while True:
    event, values = window.read()
    
    if event == sg.WIN_CLOSED or event=="Exit":
        break
    elif event == "Submit":
        dir1=values["-IN1-"]
        dir2=values["-IN2-"]
        break
